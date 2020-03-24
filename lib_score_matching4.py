#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

import numpy as np
import math
import sys
sys.path.append(".")

from lib_lanmt_modules import TransformerEncoder
from lib_lanmt_model2 import LANMTModel2
from lib_simple_encoders import ConvolutionalEncoder
from lib_lanmt_modules import TransformerCrossEncoder
from nmtlab.models import Transformer
from nmtlab.utils import OPTS


class LatentScoreNetwork4(Transformer):

    def __init__(self, lanmt_model, hidden_size=256, latent_size=8, noise=0.03):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self.set_stepwise_training(False)
        super(LatentScoreNetwork4, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        lanmt_model.train(False)
        self._lanmt = [lanmt_model]
        self.enable_valid_grad = True
        self.noise = noise
        self.train()

    def prepare(self):
        # self._encoder = ConvolutionalEncoder(None, self._hidden_size, 3)
        self._encoder = TransformerCrossEncoder(
          None, self._hidden_size, 3)
        self.lat2hid = nn.Linear(self._latent_size, self._hidden_size)
        self.hid2lat = nn.Linear(self._hidden_size, self._latent_size)
        self._hidden2energy = nn.Sequential(
            nn.Linear(self._hidden_size, 200),
            nn.ELU(),
            nn.Linear(200, 1)
        )

    def compute_energy(self, z, y_mask, x_states, x_mask):
        # z : [bsz, y_length, lat_size]
        h = self.lat2hid(z)  # [bsz, y_length, hid_size]
        energy_states = self._encoder(h, y_mask, x_states, x_mask)  # [bsz, y_length, hid_size]
        energy_states_mean = (energy_states * y_mask[:, :, None]).sum(1) / y_mask.sum(1)[:, None]  # [bsz, hid_size]
        energy = self._hidden2energy(energy_states_mean)  # [bsz, 1]
        return energy[:, 0]

    def compute_score(self, z, y_mask, x_states, x_mask):
        # z : [bsz, y_length, lat_size]
        h = self.lat2hid(z)  # [bsz, y_length, hid_size]
        score = self._encoder(h, y_mask, x_states, x_mask)  # [bsz, y_length, hid_size]
        score = self.hid2lat(score)
        return score

    def compute_targets(self, z, y_mask, x_states, x_mask, p_prob):
        lanmt = self.nmt()

        hid = lanmt.lat2hid(z)
        decoder_states = lanmt.decoder(hid, y_mask, x_states, x_mask)
        logits = lanmt.expander_nn(decoder_states)

        shape = logits.shape
        y_pred = logits.argmax(-1)
        nll = F.cross_entropy(
          logits.view(shape[0] * shape[1], -1),
          y_pred.view(shape[0] * shape[1]), reduction="none")
        nll = nll.view(shape[0], shape[1])
        logpy = (nll * y_mask).sum(1)

        mu, stddev = p_prob[..., :lanmt.latent_dim], p_prob[..., lanmt.latent_dim:]
        stddev = F.softplus(stddev)
        logpz = -0.5 * ( (z - mu) / stddev ) ** 2 - torch.log(stddev * math.sqrt(2 * math.pi))
        logpz = logpz.sum(2)
        logpz = (logpz * y_mask).sum(1)

        # outputs : [bsz,]
        return logpy, logpz

    def compute_loss(self, x, x_mask):
        lanmt = self.nmt()
        latent_dim = lanmt.latent_dim
        x_states = lanmt.embed_layer(x)
        x_states = lanmt.x_encoder(x_states, x_mask)

        x_lens = x_mask.sum(1)
        delta = lanmt.predict_length(x_states, x_mask)
        y_lens = delta + x_lens
        # y_lens = x_lens
        y_max_len = torch.max(y_lens.long()).item()
        batch_size = list(x_states.shape)[0]
        y_mask = torch.arange(y_max_len)[None, :].expand(batch_size, y_max_len).cuda()
        y_mask = (y_mask < y_lens[:, None]).float()

        pos_states = lanmt.pos_embed_layer(y_mask[:, :, None]).expand(
          list(y_mask.shape) + [lanmt.hidden_size])
        p_states = lanmt.prior_encoder(pos_states, y_mask, x_states, x_mask)
        p_prob = lanmt.p_hid2lat(p_states)
        z0 = p_prob[..., :latent_dim]

        zs = [z0]
        z = z0
        for refine_idx in range(4):
            hid = lanmt.lat2hid(z)
            decoder_states = lanmt.decoder(hid, y_mask, x_states, x_mask)
            logits = lanmt.expander_nn(decoder_states)
            y_pred = logits.argmax(-1)
            y_pred = y_pred * y_mask.long()
            y_states = lanmt.embed_layer(y_pred)
            q_states = lanmt.q_encoder_xy(y_states, y_mask, x_states, x_mask)
            q_prob = lanmt.q_hid2lat(q_states)
            z = q_prob[..., :lanmt.latent_dim]
            zs.append(z)

        if np.random.random_sample() < 0.5:
          z_ini = zs[0]
          z_fin = zs[np.random.randint(len(zs)-1)+1]
        else:
          choice = np.random.choice(len(zs), 2)
          ini, fin = choice[0], choice[1]
          ini, fin = (fin, ini) if fin < ini else (ini, fin)
          z_ini = zs[ini]
          z_fin = zs[fin]

        z_ini = z_ini + torch.randn_like(z_ini) * self.noise
        z_fin = z_fin + torch.randn_like(z_fin) * self.noise
        z_diff = (z_fin - z_ini).detach()

        with torch.no_grad():
            logpy_ini, logpz_ini = self.compute_targets(
                z_ini, y_mask, x_states, x_mask, p_prob)
            logpy_fin, logpz_fin = self.compute_targets(
                z_fin, y_mask, x_states, x_mask, p_prob)
        logpyz_ini = (logpy_ini + logpz_ini)
        logpyz_fin = (logpy_fin + logpz_fin)
        targets_diff = (logpyz_ini - logpyz_fin).detach()

        """
        energy = self.compute_energy(z_ini, y_mask, x_states, x_mask)
        dummy = torch.ones_like(energy)
        dummy.requires_grad = True

        grad = autograd.grad(
          energy,
          z_ini,
          create_graph=True,
          grad_outputs=dummy
        )

        score_match_loss = ( ( (grad[0] * z_diff) * y_mask[:, :, None] ).sum(2).sum(1) - (targets_diff) )**2
        """
        score = self.compute_score(z_ini, y_mask, x_states, x_mask)
        score_match_loss = ( ( (score * z_diff) * y_mask[:, :, None] ).sum(2).sum(1) - (targets_diff) )**2
        score_match_loss = score_match_loss.mean(0)

        # E(z, x) <- -1 * log p(y, z| x)
        return {"loss": score_match_loss}

    def forward(self, x, y, sampling=False):
        x_mask = self.to_float(torch.ne(x, 0))
        score_map = self.compute_loss(x, x_mask)
        return score_map

    def refine(self, z, x, mask=None, n_steps=50, step_size=0.001):
        if mask is not None:
            mask = mask.float()
        if not OPTS.evaluate:
            # with torch.no_grad():
            refined_z, _ = self.compute_delta_inference(x, mask, z)
        for _ in range(n_steps):
            energy, grad = self.compute_energy(z, x, mask)
            if not OPTS.evaluate:
                print((z - refined_z).norm(2).detach().cpu().numpy(),
                      energy.mean().detach().cpu().numpy(),
                      grad.norm(2).detach().cpu().numpy())
            z = z + step_size * grad
            # noise = torch.randn_like(z) * np.sqrt(step_size * 2)
            # z = z + step_size * grad + noise
            # norm = grad.norm(dim=2)
            # max_pos = norm.argmax(1)
            # if norm.max() < 0.5:
            #     break
            # z[torch.arange(z.shape[0]), max_pos] += step_size * grad[torch.arange(z.shape[0]), max_pos]
            # print(grad.norm(dim=2))
        if not OPTS.evaluate:
            raise SystemExit
        return z

    def nmt(self):
        return self._lanmt[0]

if __name__ == '__main__':
    import sys
    sys.path.append(".")
    # Testing
    lanmt = LANMTModel2(
        src_vocab_size=1000, tgt_vocab_size=1000,
        prior_layers=1, decoder_layers=1)
    snet = LatentScoreNetwork4(lanmt)
    x = torch.tensor([[1,2,3,4,5]])
    y = torch.tensor([[1,2,3]])
    if torch.cuda.is_available():
        lanmt.cuda()
        snet.cuda()
        x = x.cuda()
        y = y.cuda()
    snet(x, y)
