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
from lib_simple_encoders import ConvolutionalCrossEncoder, ConvolutionalEncoder
from nmtlab.models import Transformer
from nmtlab.utils import OPTS

from tensorboardX import SummaryWriter
from lib_envswitch import envswitch

class CorruptionBasedScoreNetwork(Transformer):
    """
    1. find target y'
    2. corrupt y' to obtain y~
    3. get z~ by running q(z|x, y~)
    4. set teacher gradient to grad_{z~} p(y'|x, z~) or (z' - z~)
    5. train score matching loss
    """

    def __init__(
        self, lanmt_model, hidden_size=256, latent_size=8,
        noise=0.1, targets="logpy", decoder="fixed", training_mode="energy", imitation=False, imit_rand_steps=1, enable_valid_grad=True):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self.imitation = imitation
        self.imit_rand_steps = imit_rand_steps
        self.training_mode = training_mode
        self._hidden_size = latent_size * 4
        self._latent_size = latent_size
        self.set_stepwise_training(False)
        lanmt_model.train(False)
        self._lanmt = [lanmt_model]
        super(LatentScoreNetwork5, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        self.enable_valid_grad = True
        self.train()
        self._mycnt = 0

        self.noise = noise
        self.targets = targets

        self.tb_str = "{}_{}_{}_{}_{}_{}".format(targets, decoder, training_mode, noise, imitation, imit_rand_steps)
        if envswitch.who() == "shu":
            main_dir = "{}/data/wmt14_ende_fair/tensorboard".format(os.getenv("HOME"))
        else:
            main_dir = "/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/tensorboard/"

        self._tb= SummaryWriter(
          log_dir="{}/{}".format(main_dir, self.tb_str), flush_secs=10)

    def prepare(self):
        self.x_embed = nn.Embedding(self.nmt()._src_vocab_size, self._hidden_size)
        self.x_encoder = ConvolutionalEncoder(self.x_embed, self._hidden_size, 3)
        # TODO (jason) replace the final MLP with a ConvNet with pooling (it gave a CuDNN error last time I tried it)
        self.lat2hid = nn.Linear(self._latent_size, self._hidden_size)
        if OPTS.ebmtype == "conv":
            self._encoder = ConvolutionalCrossEncoder(None, self._hidden_size, 3)
        else:
            self._encoder = TransformerCrossEncoder(None, self._hidden_size, 3)
        self.hid2energy = nn.Sequential(
            nn.Linear(self._hidden_size, self._hidden_size // 4),
            nn.ELU(),
            nn.Linear(self._hidden_size // 4, 1)
        )
        if OPTS.modeltype == "fakegrad":
            self.fakegrad_net = nn.Linear(self._hidden_size, 8)
        self.layernorm2 = nn.LayerNorm(self._hidden_size)
        self.layernorm3 = nn.LayerNorm(self._hidden_size)

    def compute_energy(self, z, y_mask, x_states, x_mask):
        # z : [bsz, y_length, lat_size]
        h = self.lat2hid(z)  # [bsz, y_length, hid_size]
        h = self.layernorm2(h)
        energy = self._encoder(h, y_mask, x_states, x_mask)  # [bsz, y_length, hid_size]
        energy = self.layernorm3(energy)
        if OPTS.modeltype == "fakegrad":
            return self.fakegrad_net(energy)
        energy_states_mean = (energy* y_mask[:, :, None]).sum(1) / y_mask.sum(1)[:, None]  # [bsz, hid_size]
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
            if OPTS.modeltype == "fakegrad":
                score = energy
            else:
                grad = autograd.grad(energy, z, create_graph=False, grad_outputs=dummy)
                score = grad[0].detach()
            z = z + score * lr
            lr = lr * decay
        return z

    def energy_line_search(self, z, y, y_mask, x_states, x_mask, p_prob, n_iter, c=0.5, tau=0.5):
        # z : [bsz, y_length, lat_size]
        while True:
            for idx in range(n_iter):
                z_ini = z
                targets_ini = self.compute_targets(z_ini, y, y_mask, x_states, x_mask, p_prob)
                z.requires_grad = True
                energy = self.compute_energy(z, y_mask, x_states, x_mask)
                dummy = torch.ones_like(energy)
                dummy.requires_grad = True
                grad = autograd.grad(energy, z, create_graph=False, grad_outputs=dummy)
                score = grad[0].detach()

                while True:
                    alpha = 1.0
                    z_fin = z_ini + score * alpha
                    targets_fin = self.compute_targets(z_fin, y, y_mask, x_states, x_mask, p_prob)
                    diff = targets_fin - targets_ini
                    if diff >= alpha * c:
                        z = z_fin
                        break
                    alpha *= tau
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
        logpy = (logpy * y_mask).sum(1) #/ y_mask.sum(1)
        logpy = logpy.detach()
        return logpy

    def compute_logpz(self, z, y_mask, p_prob):
        latent_dim = self.nmt().latent_dim
        p_mean, p_stddev = p_prob[..., :latent_dim], F.softplus(p_prob[..., latent_dim:])
        logpz = -0.5 * ((z - p_mean) / p_stddev) ** 2 - torch.log(p_stddev * math.sqrt(2 * math.pi))
        logpz = logpz.sum(2)
        logpz = (logpz * y_mask).sum(1) #/ y_mask.sum(1)
        logpz = logpz.detach()
        return logpz

    def compute_logqz(self, z, y, y_mask, x_states, x_mask):
        lanmt = self.nmt()
        latent_dim = self.nmt().latent_dim
        q_prob = lanmt.compute_posterior(y, y_mask, x_states, x_mask)
        q_mean, q_stddev = q_prob[..., :latent_dim], F.softplus(q_prob[..., latent_dim:])
        logqz = -0.5 * ((z - q_mean) / q_stddev) ** 2 - torch.log(q_stddev * math.sqrt(2 * math.pi))
        logqz = logqz.sum(2)
        logqz = (logqz * y_mask).sum(1) #/ y_mask.sum(1)
        logqz = logqz.detach()
        return logqz

    def compute_targets(self, z, y_mask, x_states, x_mask, p_prob, y=None):
        lanmt = self.nmt()
        latent_dim = lanmt.latent_dim
        logpy, logpz, logqz = 0, 0, 0

        logits = self.get_logits(z, y_mask, x_states, x_mask)
        if y is not None:
            y_pred = y
        else:
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

        return self.compute_targets(q_mean, y_mask, x_states, x_mask, p_prob, y=y_pred)

    def compute_loss(self, x, x_mask, y, y_mask):
        lanmt = self.nmt()
        latent_dim = lanmt.latent_dim
        with torch.no_grad():
            x_states = lanmt.embed_layer(x)
            x_states = lanmt.x_encoder(x_states, x_mask)
            x_states = x_states.detach()

            pos_states = lanmt.pos_embed_layer(y).expand(list(y_mask.shape) + [lanmt.hidden_size])
            p_states = lanmt.prior_encoder(pos_states, y_mask, x_states, x_mask)
            p_prob = lanmt.p_hid2lat(p_states).detach()

        p_mean, p_stddev = p_prob[..., :latent_dim], F.softplus(p_prob[..., latent_dim:])
        stddev = p_stddev * torch.randn_like(p_stddev)
        if self.noise == "rand":
            stddev = stddev * np.random.random_sample()
        if self.training:
            z_noise = p_mean + stddev * 0.1
        else:
            z_noise = p_mean
        z_clean = p_mean # only used for monitoring, not used for training
        z_clean.requires_grad_(True)
        z_noise.requires_grad_(True)
        ebm_x_states = self.x_encoder(x, x_mask)

        if self.imitation and self._mycnt >= 2000 and False: # Perform K SGD steps during training
            n_iter = np.random.randint(1, self.imit_rand_steps)
            z_noise = self.energy_sgd(z_noise, y_mask, x_states, x_mask, n_iter=n_iter, lr=0.10, decay=1.00)
            if n_iter > 0:
                z_noise.requires_grad = True

        z_d_noise = z_noise
        with torch.no_grad():
            for idx in range(np.random.randint(1, 2)):
                z_d_noise = self.delta_refine(z_d_noise, y_mask, x_states, x_mask)
            # z_d_clean = self.delta_refine(z_clean, y_mask, x_states, x_mask)

        z_ini, z_fin = z_noise, z_d_noise
        z_diff = (z_fin - z_ini).detach()

        # z_sgd = self.energy_sgd(z_clean, y_mask, x_states, x_mask, n_iter=4, lr=0.10, decay=1.00).detach()

        with torch.no_grad():
            targets_ini = self.compute_targets2(z_ini, y_mask, x_states, x_mask, p_prob)
            targets_fin = self.compute_targets2(z_fin, y_mask, x_states, x_mask, p_prob)

            # targets_clean = self.compute_targets2(z_clean, y_mask, x_states, x_mask, p_prob)
            # targets_d_clean = self.compute_targets2(z_d_clean, y_mask, x_states, x_mask, p_prob)

            # targets_sgd = self.compute_targets2(z_sgd, y_mask, x_states, x_mask, p_prob)

        # targets_diff_ref = (targets_d_clean - targets_clean).mean().item()
        # targets_diff_sgd = (targets_sgd - targets_clean).mean().item()
        self._mycnt += 1
        # if self._mycnt % 1 == 0:
            # self._tb.add_scalar("monitor/targets_diff_ref", targets_diff_ref, self._mycnt)
            # self._tb.add_scalar("monitor/targets_diff_sgd", targets_diff_sgd, self._mycnt)

        targets_diff = targets_fin - targets_ini
        energy = self.compute_energy(z_ini, y_mask, ebm_x_states, x_mask)
        dummy = torch.ones_like(energy)
        dummy.requires_grad = True
        if OPTS.modeltype == "fakegrad":
            score = energy
        else:
            grad = autograd.grad(energy, z_ini, create_graph=True, grad_outputs=dummy)
            score = grad[0]

        score_match_loss = ( ( (score * z_diff) * y_mask[:, :, None] ).sum(2).sum(1) - (targets_diff) )**2
        score_match_loss = score_match_loss.mean(0)
        self._tb.add_scalar("monitor/loss", score_match_loss, self._mycnt)

        return {"loss": score_match_loss, "zdiff": z_diff.mean(), "tgdiff": targets_diff.mean()}

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
            ebm_x_states = self.x_encoder(x, x_mask)
            z_ = self.energy_sgd(z, y_mask, ebm_x_states, x_mask, n_iter=n_iter, lr=lr, decay=decay).detach()

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
