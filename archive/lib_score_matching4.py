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

from tensorboardX import SummaryWriter

class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.last_average = 0.0

    def forward(self, x):
        new_average = self.mu*x + (1-self.mu)*self.last_average
        self.last_average = new_average
        return new_average


class LatentScoreNetwork4(Transformer):

    def __init__(self, lanmt_model, hidden_size=256, latent_size=8, noise=0.1, training_mode="learn_score"):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self.training_mode = training_mode
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self.set_stepwise_training(False)
        super(LatentScoreNetwork4, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        lanmt_model.train(False)
        self._lanmt = [lanmt_model]
        self.enable_valid_grad = True
        self.noise = noise
        self.train()
        self.targets_diff_ref = EMA(0.9)
        self.targets_diff_sgd = EMA(0.9)
        self._mycnt = 0
        self.tb_str = self.training_mode
        self._tb_main = SummaryWriter(
          log_dir="/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/tensorboard/{}".format(training_mode),
          filename_suffix="{}_{}".format(self.tb_str, "loss"), flush_secs=10)
        self._tb_ref = SummaryWriter(
          log_dir="/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/tensorboard/{}".format(training_mode),
          filename_suffix="{}_{}".format(self.tb_str, "ref"), flush_secs=10)
        self._tb_1step = SummaryWriter(
          log_dir="/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/tensorboard/{}/1step".format(training_mode),
          filename_suffix="{}_{}".format(self.tb_str, "1step"), flush_secs=10)
        self._tb_10steps = SummaryWriter(
          log_dir="/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/tensorboard/{}/10step".format(training_mode),
          filename_suffix="{}_{}".format(self.tb_str, "10steps"), flush_secs=10)

    def prepare(self):
        # self._encoder = ConvolutionalEncoder(None, self._hidden_size, 3)
        self._encoder = TransformerCrossEncoder(
          None, self._hidden_size, 6)
        self.lat2hid = nn.Linear(self._latent_size, self._hidden_size)
        if self.training_mode == "learn_score":
            self.hid2lat = nn.Linear(self._hidden_size, self._latent_size)
        if self.training_mode == "learn_energy":
            self.hid2energy = nn.Sequential(
                nn.Linear(self._hidden_size, 200),
                nn.ELU(),
                nn.Linear(200, 1)
            )

    def compute_energy(self, z, y_mask, x_states, x_mask):
        # z : [bsz, y_length, lat_size]
        h = self.lat2hid(z)  # [bsz, y_length, hid_size]
        energy_states = self._encoder(h, y_mask, x_states, x_mask)  # [bsz, y_length, hid_size]
        energy_states_mean = (energy_states * y_mask[:, :, None]).sum(1) / y_mask.sum(1)[:, None]  # [bsz, hid_size]
        energy = self.hid2energy(energy_states_mean)  # [bsz, 1]
        return energy[:, 0]

    def compute_score(self, z, y_mask, x_states, x_mask):
        # z : [bsz, y_length, lat_size]
        h = self.lat2hid(z)  # [bsz, y_length, hid_size]
        score = self._encoder(h, y_mask, x_states, x_mask)  # [bsz, y_length, hid_size]
        score = self.hid2lat(score)
        return score

    def energy_sgd(self, z, y_mask, x_states, x_mask, n_iter, lr):
        # z : [bsz, y_length, lat_size]
        for idx in range(n_iter):
            z = z.detach().clone()
            z.requires_grad = True
            if self.training_mode == "learn_energy":
                energy = self.compute_energy(z, y_mask, x_states, x_mask)
                dummy = torch.ones_like(energy)
                dummy.requires_grad = True

                grad = autograd.grad(
                  energy,
                  z,
                  create_graph=True,
                  grad_outputs=dummy
                )
                z = z.detach()

                score = grad[0]
            elif self.training_mode == "learn_score":
                score = self.compute_score(z, y_mask, x_states, x_mask)

            score = score.detach()
            z = z + score * lr
            lr = lr / 2.0
        return z

    def compute_targets(self, z, y_mask, x_states, x_mask, p_prob):
        lanmt = self.nmt()

        hid = lanmt.lat2hid(z)
        decoder_states = lanmt.decoder(hid, y_mask, x_states, x_mask)
        decoder_states = decoder_states.detach()
        logits = lanmt.expander_nn(decoder_states)

        shape = logits.shape
        y_pred = logits.argmax(-1)
        nll = F.cross_entropy(
          logits.view(shape[0] * shape[1], -1),
          y_pred.view(shape[0] * shape[1]), reduction="none")
        nll = nll.view(shape[0], shape[1])
        logpy = -1 * (nll * y_mask).sum(1)
        logpy = logpy.detach()

        mu, stddev = p_prob[..., :lanmt.latent_dim], p_prob[..., lanmt.latent_dim:]
        stddev = F.softplus(stddev)
        logpz = -0.5 * ( (z - mu) / stddev ) ** 2 - torch.log(stddev * math.sqrt(2 * math.pi))
        logpz = logpz.sum(2)
        logpz = (logpz * y_mask).sum(1)
        logpz = logpz.detach()

        # outputs : [bsz,]
        return logpy, logpz, logpy + logpz

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
        p_mu, p_stddev = p_prob[..., :latent_dim], F.softplus(p_prob[..., latent_dim:])
        noise = self.noise * np.random.random_sample()
        z0 = p_mu + p_stddev * torch.randn_like(p_stddev) * noise

        zs = [z0]
        z = z0
        for refine_idx in range(3):
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

        if np.random.random_sample() < 1.1:
          z_ini = zs[0]
          z_fin = zs[np.random.randint(len(zs)-1)+1]
          # z_fin = z_ini + (z_fin - z_ini) * np.random.random_sample()
        else:
          choice = np.random.choice(len(zs), 2)
          ini, fin = choice[0], choice[1]
          ini, fin = (fin, ini) if fin < ini else (ini, fin)
          z_ini = zs[ini]
          z_fin = zs[fin]

        # z_ini = z_ini + torch.randn_like(z_ini) * (np.random.random_sample() * self.noise)
        z_ini = z0
        z_fin = z_fin
        z_diff = (z_fin - z_ini).detach()

        with torch.no_grad():
            logpyz_ini, _, _ = self.compute_targets(z_ini, y_mask, x_states, x_mask, p_prob)
            logpyz_fin, _, _ = self.compute_targets(z_fin, y_mask, x_states, x_mask, p_prob)
        targets_diff = (logpyz_fin - logpyz_ini).detach()
        targets_diff_ref = self.targets_diff_ref(targets_diff.mean().item())

        z_sgd_1step = self.energy_sgd(z_ini, y_mask, x_states, x_mask, n_iter=1, lr=3.00)
        z_sgd_10steps = self.energy_sgd(z_ini, y_mask, x_states, x_mask, n_iter=5, lr=3.00)
        z_sgd_1step = z_sgd_1step.detach()
        z_sgd_10steps = z_sgd_10steps.detach()
        with torch.no_grad():
            logpyz_sgd_1step, _, _ = self.compute_targets(z_sgd_1step, y_mask, x_states, x_mask, p_prob)
            logpyz_sgd_10steps, _, _ = self.compute_targets(z_sgd_10steps, y_mask, x_states, x_mask, p_prob)
        targets_diff_sgd_1step = (logpyz_sgd_1step - logpyz_ini).mean().item()
        targets_diff_sgd_10steps = (logpyz_sgd_10steps - logpyz_ini).mean().item()
        self._mycnt += 1
        if self._mycnt % 1 == 0:
            if self.training_mode == "learn_energy":
                energy_diff_sgd_1step = self.compute_energy(z_sgd_1step, y_mask, x_states, x_mask)\
                    - self.compute_energy(z_ini, y_mask, x_states, x_mask)
                energy_diff_sgd_1step = energy_diff_sgd_1step.mean().detach()
                energy_diff_sgd_10steps = self.compute_energy(z_sgd_10steps, y_mask, x_states, x_mask)\
                    - self.compute_energy(z_ini, y_mask, x_states, x_mask)
                energy_diff_sgd_10steps = energy_diff_sgd_10steps.mean().detach()
                #print ("energy_diff : {:.3f}    targets_diff : ref {:.3f}   sgd {:.3f}     ".format(
                #  energy_diff, targets_diff_ref, targets_diff_sgd))
                self._tb_1step.add_scalar("monitor/energy_diff", energy_diff_sgd_1step, self._mycnt)
                self._tb_10steps.add_scalar("monitor/energy_diff", energy_diff_sgd_10steps, self._mycnt)
            # elif self.training_mode == "learn_score":
                #print ("targets_diff : ref {:.3f}   sgd {:.3f}     ".format(targets_diff_ref, targets_diff_sgd))
            self._tb_ref.add_scalar("monitor/targets_diff_ref", targets_diff_ref, self._mycnt)
            self._tb_1step.add_scalar("monitor/targets_diff", targets_diff_sgd_1step, self._mycnt)
            self._tb_10steps.add_scalar("monitor/targets_diff", targets_diff_sgd_10steps, self._mycnt)

        if self.training_mode == "learn_energy":
            energy = self.compute_energy(z_ini, y_mask, x_states, x_mask)
            dummy = torch.ones_like(energy)
            dummy.requires_grad = True

            grad = autograd.grad(
              energy,
              z_ini,
              create_graph=True,
              grad_outputs=dummy
            )
            score = grad[0]

        elif self.training_mode == "learn_score":
            score = self.compute_score(z_ini, y_mask, x_states, x_mask)

        score_match_loss = ( ( (score * z_diff) * y_mask[:, :, None] ).sum(2).sum(1) - (targets_diff) )**2
        score_match_loss = score_match_loss.mean(0)
        self._tb_main.add_scalar("monitor/loss", score_match_loss, self._mycnt)

        # E(z, x) <- -1 * log p(y, z| x)
        return {"loss": score_match_loss}

    def forward(self, x, y, sampling=False):
        x_mask = self.to_float(torch.ne(x, 0))
        score_map = self.compute_loss(x, x_mask)
        return score_map

    def translate(self, x, n_iter=0, lr=0.1):
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
        y_mask = torch.arange(y_max_len)[None, :].expand(batch_size, y_max_len)
        y_mask = (y_mask < y_lens[:, None])
        y_mask = y_mask.float()
        # y_mask = x_mask

        # Compute p(z|x)
        pos_states = lanmt.pos_embed_layer(y_mask[:, :, None]).expand(
          list(y_mask.shape) + [lanmt.hidden_size])
        p_states = lanmt.prior_encoder(pos_states, y_mask, x_states, x_mask)
        p_prob = lanmt.p_hid2lat(p_states)
        z = p_prob[..., :lanmt.latent_dim]
        z_ = self.energy_sgd(z, y_mask, x_states, x_mask, n_iter=5, lr=3.0)
        """
        _, _, t1 = self.compute_targets(z, y_mask, x_states, x_mask, p_prob)
        _, _, t2 = self.compute_targets(z_, y_mask, x_states, x_mask, p_prob)
        print ( "{:.2f} ".format((t2-t1).mean()),  )
        """

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
