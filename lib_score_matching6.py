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
from contextlib import suppress
import sys
sys.path.append(".")

from lib_ebm_modules import ConvNetEnergyFn, EnergyFn, ScoreFn, cosine_loss_tc, score_matching_loss
from lib_lanmt_model2 import LANMTModel2
from lib_corrpution import random_token_corruption
from nmtlab.models import Transformer
from nmtlab.utils import OPTS

from lib_envswitch import envswitch

class LatentScoreNetwork6(Transformer):

    def __init__(
        self, lanmt_model, hidden_size=256, latent_size=8,
        noise=1.0, train_sgd_steps=0, train_step_size=0.0, train_delta_steps=2,
        modeltype="realgrad",
        ebm_useconv=False, direction_n_layers=4, magnitude_n_layers=4,
        enable_valid_grad=True):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self.set_stepwise_training(False)

        self._mycnt = 0
        self.noise = noise
        self.train_sgd_steps = train_sgd_steps
        self.train_step_size = train_step_size
        self.train_delta_steps = train_delta_steps
        self.modeltype = modeltype
        self.ebm_useconv = ebm_useconv
        self.direction_n_layers = direction_n_layers
        self.magnitude_n_layers = magnitude_n_layers
        assert self.modeltype in ["realgrad", "fakegrad"]

        super(LatentScoreNetwork6, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        lanmt_model.train(False)
        self._lanmt = [lanmt_model]
        self.enable_valid_grad = True
        self.train()

    def prepare(self):
        D_lat, D_hid = self._latent_size, self._hidden_size
        energy_cls = ConvNetEnergyFn if self.ebm_useconv else EnergyFn
        # only keeping for backwards compatibility for EBM models trained with magnitude (e.g. IWSLT)
        self.magnitude_fn = energy_cls(
            D_lat, D_hid, n_layers=self.magnitude_n_layers, positive=True) # Learn z_diff.norm(1, 2)
        if self.modeltype == "realgrad": # Learn the energy function
            self.score_fn = energy_cls(
                D_lat, D_hid, n_layers=self.direction_n_layers, positive=False)
        elif self.modeltype == "fakegrad": # Directly learn the gradient of the energy
            self.score_fn = ScoreFn(
                D_lat, D_hid, D_lat, n_layers=self.direction_n_layers, positive=False)

    def energy_sgd_batch(self, z, y_mask, x_states, x_mask, n_iter, step_size):
        # z : [bsz, y_length, lat_size]
        y_length = y_mask.sum(1).float()
        z_list, score_list = [], []
        for idx in range(n_iter):
            z = z.detach().clone()
            z.requires_grad = True

            score = self.score_fn.score(z, y_mask, x_states, x_mask, create_graph=True).detach()
            score = score * step_size
            score_list.append(score)

            z = z + score
            z_list.append(z)
        return z_list, score_list

    def energy_sgd(self, z, y_mask, x_states, x_mask, n_iter, step_size):
        # z : [bsz, y_length, lat_size]
        y_length = y_mask.sum(1).float()
        for idx in range(n_iter):
            z = z.detach().clone()
            z.requires_grad = True

            score = self.score_fn.score(z, y_mask, x_states, x_mask, create_graph=True).detach()
            score = score * step_size

            z = z + score
        return z

    def energy_adaptive(self, z, y_mask, x_states, x_mask, step_size, norm_thresh=1/16, max_iters=6):
        # z : [bsz, y_length, lat_size]
        y_length = y_mask.sum(1).float()
        len_int = int(y_length.item())
        score_norm_prev = None # norm in 1st iteration (not prev, but 1st refinement step)
        n_iter = 0
        while True:
            z = z.detach().clone()
            z.requires_grad = True

            score = self.score_fn.score(z, y_mask, x_states, x_mask, create_graph=True).detach()
            score_norm = (score * y_mask[:, :, None]) ** 2
            #score_norm = (score ** 2) * y_mask[:, :, None]
            score_norm = score_norm.sum(2).sqrt().sum(1) / y_length
            #score_norm = score_norm.sum(2).sum(1).sqrt() / y_length

            z = z + score * step_size
            n_iter += 1

            if not (score_norm_prev is None) and (score_norm / score_norm_prev).item() < norm_thresh:
                break
            if n_iter >= max_iters:
                break
            if score_norm_prev is None:
                score_norm_prev = score_norm

        return z

    def compute_targets(self, z, y_mask, x_states, x_mask, p_prob, y_pred=None):
        lanmt = self.nmt()
        latent_dim = lanmt.latent_dim
        logpy, logpz, logqz = 0, 0, 0

        logits = lanmt.get_logits(z, y_mask, x_states, x_mask)
        if y_pred == None:
            y_pred = logits.argmax(-1)
            y_pred = y_pred * y_mask.long()
        logpy = lanmt.compute_logpy(logits, y_pred, y_mask)

        #if self.targets == "joint" or self.targets == "elbo":
        #    logpz = lanmt.compute_logpz(z, y_mask, p_prob)

        #if self.targets == "elbo":
        #    logqz = lanmt.compute_logqz(z, y_pred, y_mask, x_states, x_mask)

        #targets = logpy + logpz - logqz
        # NOTE using only log(py) to do line search for now
        targets = (logpy * y_mask).sum(1) / y_mask.sum(1)
        return targets

    def energy_line_search(self, z, y_mask, x_states, x_mask, p_prob, n_iter):
        # z : [bsz, y_length, lat_size]
        latent_dim = self.nmt().latent_dim
        targets_length = y_mask.shape[1]

        for iter_idx in range(n_iter):
            z_ini = z.detach().clone()
            #with torch.no_grad():
            #    targets_ini = self.compute_targets(z_ini, y_mask, x_states, x_mask, p_prob)
            z_ini.requires_grad = True
            score = self.score_fn.score(z_ini, y_mask, x_states, x_mask, create_graph=True).detach()

            with torch.no_grad():
                #alphas = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
                alphas = [0.5, 1.0, 1.5]
                targets = []
                z_fins = []
                for alpha in alphas:
                    z_fin = z_ini + score * alpha # [batch_size, targets_length, latent_size]
                    targets_fin = self.compute_targets(z_fin, y_mask, x_states, x_mask, p_prob) # [batch_size]
                    targets.append(targets_fin)
                    z_fins.append(z_fin)
                targets = torch.stack(targets, dim=0) # [num_alphas, batch_size]
                z_fins = torch.stack(z_fins, dim=0) # [num_alphas, batch_size, targets_length, latent_size]
                _, idx = targets.max(dim=0, keepdim=True) # [1, batch_size]
                idx_large = idx[:, :, None, None].repeat(1, 1, targets_length, latent_dim)
                z = z_fins.gather(dim=0, index=idx_large)[0]
                #print (alphas[idx], )
        return z

    def euc_distance(self, z1, z2, mask):
        distance = ((z2 - z1)**2).sum(2)
        distance = (distance * mask).sum(1) / mask.sum(1)
        return distance.mean()

    def compute_loss(self, x, x_mask, y, y_mask):
        lanmt = self.nmt()
        latent_dim = lanmt.latent_dim
        x_states = lanmt.embed_layer(x)
        x_states = lanmt.x_encoder(x_states, x_mask)
        y_length = y_mask.sum(1).float()

        p_prob = lanmt.compute_prior(y_mask, x_states, x_mask)
        p_mean, p_stddev = p_prob[..., :latent_dim], F.softplus(p_prob[..., latent_dim:])
        z_ini = p_mean
        if self.training:
            stddev = p_stddev * torch.randn_like(p_stddev)
            stddev = stddev * np.random.random_sample() * self.noise
            z_ini += stddev
            if self.train_sgd_steps > 0 and self._mycnt > 100000 and np.random.random_sample() < 0.5:
                with torch.no_grad() if self.modeltype == "fakegrad" else suppress():
                    z_ini = self.energy_sgd(
                        z_ini, y_mask, x_states, x_mask,
                        n_iter=self.train_sgd_steps, step_size=self.train_step_size)
        with torch.no_grad():
            z_fin = lanmt.delta_refine(
                z_ini, y_mask, x_states, x_mask, n_iter=self.train_delta_steps)

        z_ini = z_ini.detach().clone()
        z_ini.requires_grad = True

        z_diff = (z_fin - z_ini).detach() # [batch_size, targets_length, latent_size]
        if OPTS.corrupt:
            with torch.no_grad():
                # logits = lanmt.get_logits(z_fin, y_mask, x_states, x_mask)
                # y_delta = logits.argmax(-1)
                y_noise, _ = random_token_corruption(y, self._tgt_vocab_size, 0.2)
                z_noise = self.nmt().compute_posterior(y_noise, y_mask, x_states, x_mask)
                z_fin = self.nmt().compute_posterior(y, y_mask, x_states, x_mask)
                z_fin = z_fin[:, :, :latent_dim]
                stddev = z_noise[:, :, latent_dim:]
                z_ini = z_noise[:, :, :latent_dim] + stddev * torch.randn_like(stddev)
                z_diff = (z_fin - z_ini).detach()
            z_ini.requires_grad_(True)

        # [batch_size, targets_length, latent_size]
        score = self.score_fn.score(z_ini, y_mask, x_states, x_mask)
        # [batch_size]
        loss_direction = score_matching_loss(score, z_diff, y_mask)

        loss = loss_direction
        loss = loss.mean(0)
        score_map = {"loss": loss}

        if not self.training:
            score_map["cosine_sim"] = 1 - cosine_loss_tc(score, z_diff, y_mask).mean()
        return score_map

    def forward(self, x, y, sampling=False):
        x_mask = self.to_float(torch.ne(x, 0))
        y_mask = self.to_float(torch.ne(y, 0))
        score_map = self.compute_loss(x, x_mask, y, y_mask)
        self._mycnt += 1 # pretty hacky I know, sorry haha
        return score_map

    def translate(self, x, n_iter, step_size, y_mask=None, line_search=False):
        """ Testing codes.
        """
        lanmt = self.nmt()
        x_mask = lanmt.to_float(torch.ne(x, 0))
        x_states = lanmt.embed_layer(x)
        x_states = lanmt.x_encoder(x_states, x_mask)

        # Predict length
        if y_mask is None:
            x_lens = x_mask.sum(1)
            delta = lanmt.predict_length(x_states, x_mask)
            # NOTE experimental
            # delta = delta + (x_lens * 0.02).long() # IWSLT DEEN
            # delta = delta + (x_lens * 0.06).long() # WMT ROEN
            y_lens = delta.long() + x_lens.long()
            # y_lens = x_lens
            y_max_len = torch.max(y_lens.long()).item()
            batch_size = list(x_states.shape)[0]
            y_mask = torch.arange(y_max_len)[None, :].expand(batch_size, y_max_len).cuda()
            y_mask = (y_mask < y_lens[:, None])
            y_mask = y_mask.float()

        # Compute p(z|x)
        with torch.no_grad():
            p_prob = lanmt.compute_prior(y_mask, x_states, x_mask)
        z = p_prob[..., :lanmt.latent_dim]
        if OPTS.Twithout_ebm:
            z = z
        else:
            with torch.no_grad() if self.modeltype == "fakegrad" else suppress():
                if line_search:
                    z = self.energy_line_search(z, y_mask, x_states, x_mask, None, n_iter=n_iter)
                else:
                    z = self.energy_sgd(z, y_mask, x_states, x_mask, n_iter=n_iter, step_size=step_size)
                    #z = self.energy_adaptive(z, y_mask, x_states, x_mask, step_size=step_size)

        with torch.no_grad():
            logits = lanmt.get_logits(z, y_mask, x_states, x_mask)
        y_pred = logits.argmax(-1)
        y_pred = y_pred * y_mask.long()
        return y_pred, z, y_mask

    def nmt(self):
        return self._lanmt[0]

if __name__ == '__main__':
    import sys
    sys.path.append(".")
    # Testing
    lanmt = LANMTModel2(
        src_vocab_size=1000, tgt_vocab_size=1000,
        prior_layers=1, decoder_layers=1)
    snet = LatentScoreNetwork6(lanmt)
    x = torch.tensor([[1,2,3,4,5]])
    y = torch.tensor([[1,2,3]])
    if torch.cuda.is_available():
        lanmt.cuda()
        snet.cuda()
        x = x.cuda()
        y = y.cuda()
    snet(x, y)
