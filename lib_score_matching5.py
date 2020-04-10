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
import os
import sys
sys.path.append(".")

from lib_lanmt_modules import TransformerEncoder
from lib_lanmt_model2 import LANMTModel2
from lib_lanmt_modules import TransformerCrossEncoder
from lib_simple_encoders import ConvolutionalCrossEncoder
from nmtlab.models import Transformer
from nmtlab.utils import OPTS

from tensorboardX import SummaryWriter
from lib_envswitch import envswitch

class LatentScoreNetwork5(Transformer):

    def __init__(
        self, lanmt_model, hidden_size=256, latent_size=8,
        noise=0.1, targets="logpy", decoder="fixed", imitation=False, line_search_c=0.1, imit_rand_steps=1, enable_valid_grad=True):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self.imitation = imitation
        self.imit_rand_steps = imit_rand_steps
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self.set_stepwise_training(False)
        super(LatentScoreNetwork5, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        lanmt_model.train(False)
        self._lanmt = [lanmt_model]
        self.enable_valid_grad = True
        self.train()
        self._mycnt = 0

        self.noise = noise
        self.targets = targets
        self.line_search_c = line_search_c

        self.tb_str = "{}_{}_{}_{}_{}_{}".format(targets, decoder, noise, imitation, line_search_c, imit_rand_steps)
        if envswitch.who() == "shu":
            main_dir = "{}/data/wmt14_ende_fair/tensorboard".format(os.getenv("HOME"))
        else:
            main_dir = "/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/tensorboard/"

        self._tb= SummaryWriter(
          log_dir="{}/{}".format(main_dir, self.tb_str), flush_secs=10)

    def prepare(self):
        # TODO (jason) replace the final MLP with a ConvNet with pooling (it gave a CuDNN error last time I tried it)
        self.lat2hid = nn.Linear(self._latent_size, self._hidden_size)
        self._encoder = TransformerCrossEncoder(None, self._hidden_size, 3)
        self.hid2energy = nn.Sequential(
            nn.Linear(self._hidden_size, 200),
            nn.ELU(),
            nn.Linear(200, 1)
        )

    def compute_energy(self, z, y_mask, x_states, x_mask):
        # z : [bsz, y_length, lat_size]
        h = self.lat2hid(z)  # [bsz, y_length, hid_size]
        energy = self._encoder(h, y_mask, x_states, x_mask)  # [bsz, y_length, hid_size]
        energy_states_mean = (energy * y_mask[:, :, None]).sum(1) / y_mask.sum(1)[:, None]  # [bsz, hid_size]
        energy = self.hid2energy(energy_states_mean)  # [bsz, 1]
        return energy[:, 0]

    def delta_refine(self, z, y_mask, x_states, x_mask):
        lanmt = self.nmt()
        hid = lanmt.lat2hid(z)
        decoder_states = lanmt.decoder(hid, y_mask, x_states, x_mask)
        logits = lanmt.expander_nn(decoder_states)
        y_pred = logits.argmax(-1)
        y_pred = y_pred * y_mask.long()
        y_states = lanmt.embed_layer(y_pred)
        q_states = lanmt.q_encoder_xy(y_states, y_mask, x_states, x_mask)
        q_prob = lanmt.q_hid2lat(q_states)
        z = q_prob[..., :lanmt.latent_dim]
        return z

    def energy_sgd(self, z, y_mask, x_states, x_mask, n_iter, lr, decay):
        # z : [bsz, y_length, lat_size]
        for idx in range(n_iter):
            z = z.detach().clone()
            z.requires_grad = True
            energy = self.compute_energy(z, y_mask, x_states, x_mask)
            dummy = torch.ones_like(energy)
            dummy.requires_grad = True

            grad = autograd.grad(energy, z, create_graph=False, grad_outputs=dummy)
            score = grad[0].detach()
            z = z + score * lr # NOTE : We do gradient *ascent, as energy approximate ELBO.
            lr = lr * decay
        return z

    def energy_line_search(self, z, y_mask, x_states, x_mask, p_prob, n_iter, c=0.05, tau=0.5):
        # z : [bsz, y_length, lat_size]
        for idx in range(n_iter):
            z_ini = z.detach().clone()
            with torch.no_grad(): # targets are normalized by trg sequence length
                targets_ini = self.compute_targets(
                    z_ini, y_mask, x_states, x_mask, p_prob)
            z_ini.requires_grad = True
            energy = self.compute_energy(z_ini, y_mask, x_states, x_mask)
            dummy = torch.ones_like(energy)
            dummy.requires_grad = True

            grad = autograd.grad(energy, z_ini, create_graph=False, grad_outputs=dummy)
            score = grad[0].detach()

            with torch.no_grad(): # targets are normalized by trg sequence length
                alpha = 2.0
                while True:
                    z_fin = z_ini + score * alpha
                    targets_fin = self.compute_targets(
                        z_fin, y_mask, x_states, x_mask, p_prob)
                    diff = (targets_fin - targets_ini).mean().item()
                    if diff >= alpha * c or alpha <= 0.2:
                        z = z_fin
                        break
                    alpha = alpha / 2.0
        return z

    def get_logits(self, z, y_mask, x_states, x_mask):
        lanmt = self.nmt()
        hid = lanmt.lat2hid(z)
        decoder_states = lanmt.decoder(hid, y_mask, x_states, x_mask).detach()
        logits = lanmt.expander_nn(decoder_states)
        return logits

    def compute_logpy(self, logits, y, y_mask, x_states, x_mask):
        shape = logits.shape
        nll = F.cross_entropy(
          logits.view(shape[0] * shape[1], -1),
          y.view(shape[0] * shape[1]), reduction="none", ignore_index=0)
        logpy = -1 * nll
        logpy = logpy.view(shape[0], shape[1])
        logpy = (logpy * y_mask).sum(1) / y_mask.sum(1)
        logpy = logpy.detach()
        return logpy

    def compute_logpz(self, z, y_mask, p_prob):
        latent_dim = self.nmt().latent_dim
        p_mean, p_stddev = p_prob[..., :latent_dim], F.softplus(p_prob[..., latent_dim:])
        logpz = -0.5 * ( (z - p_mean) / p_stddev ) ** 2 - torch.log(p_stddev * math.sqrt(2 * math.pi))
        logpz = logpz.sum(2)
        logpz = (logpz * y_mask).sum(1) / y_mask.sum(1)
        logpz = logpz.detach()
        return logpz

    def compute_logqz(self, z, y, y_mask, x_states, x_mask):
        lanmt = self.nmt()
        latent_dim = self.nmt().latent_dim
        q_prob = lanmt.compute_posterior(y, y_mask, x_states, x_mask)
        q_mean, q_stddev = q_prob[..., :latent_dim], F.softplus(q_prob[..., latent_dim:])
        logqz = -0.5 * ( (z - q_mean) / q_stddev ) ** 2 - torch.log(q_stddev * math.sqrt(2 * math.pi))
        logqz = logqz.sum(2)
        logqz = (logqz * y_mask).sum(1) / y_mask.sum(1)
        logqz = logqz.detach()
        return logqz

    def compute_targets(self, z, y_mask, x_states, x_mask, p_prob):
        lanmt = self.nmt()
        latent_dim = lanmt.latent_dim
        logpy, logpz, logqz = 0, 0, 0

        logits = self.get_logits(z, y_mask, x_states, x_mask)
        y_pred = logits.argmax(-1)
        logpy = self.compute_logpy(logits, y_pred, y_mask, x_states, x_mask)
        logits = None

        if self.targets == "joint" or self.targets == "elbo":
            logpz = self.compute_logpz(z, y_mask, p_prob)

        if self.targets == "elbo":
            logqz = self.compute_logqz(z, y_pred, y_mask, x_states, x_mask)

        return logpy + logpz - logqz

    def compute_targets2(self, z, y_mask, x_states, x_mask, p_prob):
        lanmt = self.nmt()
        latent_dim = lanmt.latent_dim

        logits = self.get_logits(z, y_mask, x_states, x_mask)
        y_pred = logits.argmax(-1)
        logits = None
        q_prob = lanmt.compute_posterior(y_pred, y_mask, x_states, x_mask)
        q_mean = q_prob[..., :latent_dim]

        return self.compute_targets(q_mean, y_mask, x_states, x_mask, p_prob)

    def compute_loss(self, x, x_mask, y, y_mask):
        self._mycnt += 1
        lanmt = self.nmt()
        latent_dim = lanmt.latent_dim
        x_states = lanmt.embed_layer(x)
        x_states = lanmt.x_encoder(x_states, x_mask)

        p_prob = lanmt.compute_prior(y_mask, x_states, x_mask)
        p_mean, p_stddev = p_prob[..., :latent_dim], F.softplus(p_prob[..., latent_dim:])
        stddev = p_stddev * torch.randn_like(p_stddev)
        if self.noise == "rand":
            stddev = stddev * np.random.random_sample()
        z_ini = p_mean + stddev

        if self.imitation: # Perform K SGD steps during training
            n_iter = np.random.randint(0, self.imit_rand_steps)
            #z_ini = self.energy_line_search(
            #    z_ini, y_mask, x_states, x_mask, p_prob, n_iter=n_iter, c=self.line_search_c).detach()
            z_ini = self.energy_sgd(
                z_ini, y_mask, x_states, x_mask, n_iter=n_iter, lr=0.1, decay=1.0).detach()
            z_ini = z_ini.detach().clone()
            z_ini.requires_grad = True

        z_fin = self.delta_refine(z_ini, y_mask, x_states, x_mask)
        z_diff = (z_fin - z_ini).detach() # [bsz, y_length, lat_size]

        with torch.no_grad(): # targets are normalized by trg sequence length
            targets_ini = self.compute_targets(z_ini, y_mask, x_states, x_mask, p_prob) # [bsz]
            targets_fin = self.compute_targets(z_fin, y_mask, x_states, x_mask, p_prob) # [bsz]
            targets_diff = targets_fin - targets_ini # [bsz]

        energy = self.compute_energy(z_ini, y_mask, x_states, x_mask) # [bsz]
        dummy = torch.ones_like(energy)
        dummy.requires_grad = True
        grad = autograd.grad(energy, z_ini, create_graph=True, grad_outputs=dummy)
        score = grad[0] # [bsz, y_length, lat_size]

        rop = ( (score * z_diff).sum(2) * y_mask ).sum(1) / y_mask.sum(1) # normalize by the trg sentence length
        loss = (rop - targets_diff) ** 2
        loss = loss.mean(0)
        self._tb.add_scalar("monitor/loss", loss, self._mycnt)

        #if not model.training:
        if self._mycnt % 50 == 0:
            z_clean = p_mean # only used for monitoring, not used for training
            z_d_clean = self.delta_refine(z_clean, y_mask, x_states, x_mask)
            #z_sgd = self.energy_sgd(z_clean, y_mask, x_states, x_mask, n_iter=2, lr=0.10, decay=1.00).detach()
            z_sgd = self.energy_line_search(
                z_clean, y_mask, x_states, x_mask, p_prob, n_iter=20, c=self.line_search_c).detach()
            with torch.no_grad():
                targets_clean = self.compute_targets(z_clean, y_mask, x_states, x_mask, p_prob)
                targets_d_clean = self.compute_targets(z_d_clean, y_mask, x_states, x_mask, p_prob)
                targets_sgd = self.compute_targets(z_sgd, y_mask, x_states, x_mask, p_prob)

            targets_diff_ref = (targets_d_clean - targets_clean).mean().item()
            targets_diff_sgd = (targets_sgd - targets_clean).mean().item()
            self._tb.add_scalar("monitor/targets_diff_ref", targets_diff_ref, self._mycnt)
            self._tb.add_scalar("monitor/targets_diff_sgd", targets_diff_sgd, self._mycnt)

        loss = loss * 100
        return {"loss": loss}

    def forward(self, x, y, sampling=False):
        x_mask = self.to_float(torch.ne(x, 0))
        y_mask = self.to_float(torch.ne(y, 0))
        score_map = self.compute_loss(x, x_mask, y, y_mask)
        return score_map

    def translate(self, x, n_iter, lr, decay):
        """ Testing codes.
        """
        lanmt = self.nmt()
        x_mask = lanmt.to_float(torch.ne(x, 0))
        x_states = lanmt.embed_layer(x)
        x_states = lanmt.x_encoder(x_states, x_mask)

        # Predict length
        x_lens = x_mask.sum(1)
        delta = lanmt.predict_length(x_states, x_mask)
        y_lens = delta + x_lens
        # y_lens = x_lens
        y_max_len = torch.max(y_lens.long()).item()
        batch_size = list(x_states.shape)[0]
        y_mask = torch.arange(y_max_len)[None, :].expand(batch_size, y_max_len).cuda()
        y_mask = (y_mask < y_lens[:, None])
        y_mask = y_mask.float()
        # y_mask = x_mask

        # Compute p(z|x)
        pos_states = lanmt.pos_embed_layer(y_mask[:, :, None]).expand(
          list(y_mask.shape) + [lanmt.hidden_size])
        p_states = lanmt.prior_encoder(pos_states, y_mask, x_states, x_mask)
        p_prob = lanmt.p_hid2lat(p_states)
        z = p_prob[..., :lanmt.latent_dim]
        if OPTS.Twithout_ebm:
            z_ = z
        else:
            z_ = self.energy_sgd(z, y_mask, x_states, x_mask, n_iter=n_iter, lr=lr, decay=decay).detach()

        hid = lanmt.lat2hid(z_)
        decoder_states = lanmt.decoder(hid, y_mask, x_states, x_mask)
        logits = lanmt.expander_nn(decoder_states)
        y_pred = logits.argmax(-1)
        y_pred = y_pred * y_mask.long()
        # y_pred = y_pred * x_mask.long()

        return y_pred

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
