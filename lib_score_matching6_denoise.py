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
from lib_simple_encoders import ConvolutionalEncoder, ConvolutionalCrossEncoder
from lib_corrpution import random_token_corruption
from nmtlab.models import Transformer
from nmtlab.utils import OPTS

from lib_envswitch import envswitch

class EnergyFn(nn.Module):
    def __init__(self, D_lat, D_hid, dropout_ratio=0.1, D_mlp_hid=100, n_layers=3, positive=False):
        super(EnergyFn, self).__init__()
        self.lat2hid = nn.Linear(
            D_lat, D_hid)
        self.transformer = TransformerCrossEncoder(
            embed_layer=None, size=D_hid, n_layers=n_layers, dropout_ratio=dropout_ratio)
        self.conv = ConvolutionalEncoder(
            embed_layer=None, size=D_hid, n_layers=n_layers, dropout_ratio=dropout_ratio,
            cross_attention=False, skip_connect=False)
        self.hid2energy = nn.Sequential(
            nn.Dropout(p=dropout_ratio),
            nn.Linear(D_hid, D_mlp_hid),
            nn.ELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(D_mlp_hid, 1)
        )
        self.positive = positive

    def forward(self, z, y_mask, x_states, x_mask):
        # input : [batch_size, targets_length, latent_size]
        # output : [batch_size]
        h = self.lat2hid(z)
        energy = self.transformer(h, y_mask, x_states, x_mask)
        energy = energy * y_mask[:, :, None]
        energy = self.conv(energy, y_mask)
        energy = energy + (1 - y_mask)[:, :, None] * (-999999.)
        energy = energy.max(dim=1)[0]
        energy = self.hid2energy(energy)
        if self.positive:
            energy = F.softplus(energy)
        return energy[:, 0]

    def score(self, z, y_mask, x_states, x_mask, create_graph=True):
        energy = self.forward(z, y_mask, x_states, x_mask)
        dummy = torch.ones_like(energy)
        dummy.requires_grad = True
        grad = autograd.grad(energy, z, create_graph=create_graph, grad_outputs=dummy)
        score = grad[0] # [bsz, length, latent_dim]
        return score

class ScoreFn(nn.Module):
    def __init__(self, D_lat, D_hid, D_fin, dropout_ratio=0.1, D_mlp_hid=100, n_layers=4, positive=False):
        super(ScoreFn, self).__init__()
        self.lat2hid = nn.Linear(
            D_lat, D_hid)
        self.transformer = ConvolutionalCrossEncoder(
            embed_layer=None, size=D_hid, n_layers=n_layers, dropout_ratio=dropout_ratio)
        self.hid2score = nn.Linear(
            D_hid, D_fin)
        self.positive = positive

    def forward(self, z, y_mask, x_states, x_mask):
        # input : [batch_size, targets_length, latent_size]
        # output : [batch_size, targets_length, final_size]
        h = self.lat2hid(z)
        score = self.transformer(h, y_mask, x_states, x_mask)
        score = self.hid2score(score)
        if self.positive:
            score = F.softplus(score)
        return score * y_mask[:, :, None]

    def score(self, z, y_mask, x_states, x_mask):
        return self.forward(z, y_mask, x_states, x_mask)

class LatentScoreNetwork6(Transformer):

    def __init__(
        self, lanmt_model, hidden_size=256, latent_size=8,
        noise=0.1, targets="logpy", decoder="fixed", imitation=False,
        imit_rand_steps=1, cosine="T", refine_from_mean=False,
        modeltype="realgrad", enable_valid_grad=True):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self.set_stepwise_training(False)

        self._mycnt = 0
        self.imitation = imitation
        self.imit_rand_steps = imit_rand_steps
        self.cosine = cosine
        self.noise = noise
        self.refine_from_mean = refine_from_mean
        self.modeltype = modeltype
        assert self.modeltype in ["realgrad", "fakegrad"]
        assert cosine in ["C", "TC"]
        # TC : maximize cosine_sim(z_diff, score, dims=[1,2])
        #  C : maximize cosine_sim(z_diff, score1, dims=[2])
        #      maximize cosine_sim(z_diff.norm(2), score2, dims=[1])

        lanmt_model.train(False)
        self._lanmt = [lanmt_model]
        super(LatentScoreNetwork6, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        self.enable_valid_grad = True
        self.train()

    def prepare(self):
        D_lat, D_hid = self._latent_size, self._hidden_size
        if self.modeltype == "realgrad": # Learn the energy function
            self.score_fn = EnergyFn(D_lat, D_hid, positive=False)
        elif self.modeltype == "fakegrad": # Directly learn the gradient of the energy
            self.score_fn = ScoreFn(D_lat, D_hid, D_lat, positive=False)

        if self.cosine == "C":
            # Learn z_diff.norm(2)
            self.score_fn2 = ScoreFn(D_lat, D_hid, 1, positive=True)
            self.cosine_loss = cosine_loss_c
        elif self.cosine == "TC":
            self.cosine_loss = cosine_loss_tc

        self.magnitude_fn = EnergyFn(D_lat, D_hid, positive=True) # Learn z_diff.norm(1, 2)
        self.expander_nn = nn.Linear(256, self.nmt()._tgt_vocab_size)
        self.decoder = ConvolutionalEncoder(None, 256, n_layers=3)
        self.lat2hid = nn.Linear(self._latent_size, 256)

    def delta_refine(self, z, y_mask, x_states, x_mask, n_iter=1):
        lanmt = self.nmt()
        for idx in range(n_iter):
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

    def energy_sgd(self, z, y_mask, x_states, x_mask, n_iter):
        # z : [bsz, y_length, lat_size]
        y_length = y_mask.sum(1).float()
        lrs = [1.0, 1.0, 1.0]
        for idx in range(n_iter):
            z = z.detach().clone()
            z.requires_grad = True

            # magnitude = self.magnitude_fn(z, y_mask, x_states, x_mask) # [bsz]

            score = self.score_fn.score(z, y_mask, x_states, x_mask).detach()
            # score_norm = (score ** 2) * y_mask[:, :, None]
            # score_norm = score_norm.sum(2).sum(1).sqrt()
            # multiplier = magnitude * y_length.sqrt() / score_norm
            # multiplier = y_length.sqrt() / score_norm
            # score = score * multiplier[:, None, None] * lrs[idx]

            z = z + score
        return z

    def energy_line_search(self, z, y_mask, x_states, x_mask, p_prob, n_iter, c=0.05, tau=0.5):
        # implementation from https://en.wikipedia.org/wiki/Backtracking_line_search
        # hyperparameters : alpha, c, tau taken from the wiki page.
        # z : [bsz, y_length, lat_size]
        for idx in range(n_iter):
            z_ini = z.detach().clone()
            with torch.no_grad():
                targets_ini = self.compute_targets(
                    z_ini, y_mask, x_states, x_mask, p_prob)
            z_ini.requires_grad = True
            energy = self.compute_energy(z_ini, y_mask, x_states, x_mask)
            dummy = torch.ones_like(energy)
            dummy.requires_grad = True

            grad = autograd.grad(energy, z_ini, create_graph=False, grad_outputs=dummy)
            score = grad[0].detach()

            with torch.no_grad():
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
        decoder_states = lanmt.decoder(hid, y_mask, x_states, x_mask)
        logits = lanmt.expander_nn(decoder_states)
        return logits

    def cosine_loss_tc(x1, x2, mask):
        # Compute cosine loss over T and C dimension (ignoring PAD)
        # input : [batch_size, targets_length, latent_size]
        # output : [batch_size]
        x1_norm = (x1 ** 2) * mask[:, :, None]
        x1_norm = x1_norm.sum(2).sum(1).sqrt()
        x2_norm = (x2 ** 2) * mask[:, :, None]
        x2_norm = x2_norm.sum(2).sum(1).sqrt()
        num = (x1 * x2) * mask[:, :, None]
        num = num.sum(2).sum(1)
        denom = x1_norm * x2_norm
        sim = num / denom

        return 1 - sim  # [bsz]

    def euc_distance(self, z1, z2, mask):
        distance = ((z2 - z1)**2).sum(2)
        distance = (distance * mask).sum(1) / mask.sum(1)
        return distance.mean()

    def xent_loss(self, logits, y, y_mask):
        bsize, ylen = y.shape
        loss_mat = F.cross_entropy(logits.view(bsize * ylen, -1), y.flatten(), reduction="none").view(bsize, ylen)
        loss = (loss_mat * y_mask).sum() / y_mask.sum()
        return loss

    def compute_loss(self, x, x_mask, y, y_mask):
        lanmt = self.nmt()
        latent_dim = lanmt.latent_dim
        x_states = lanmt.embed_layer(x)
        x_states = lanmt.x_encoder(x_states, x_mask)
        y_length = y_mask.sum(1).float()
        assert OPTS.modeltype =="fakegrad"
        # p_prob = lanmt.compute_prior(y_mask, x_states, x_mask)
        # p_mean, p_stddev = p_prob[..., :latent_dim], F.softplus(p_prob[..., latent_dim:])
        # z_ini = p_mean
        # if self.training
        #     stddev = p_stddev * torch.randn_like(p_stddev)
        #     if self.noise == "rand":
        #         stddev = stddev * np.random.random_sample()
        #     z_ini += stddev
        #
        # with torch.no_grad():
        #     z_delta = self.delta_refine(z_ini, y_mask, x_states, x_mask, n_iter=2)
        #     if OPTS.fin == "y":
        #         z_fin = self.nmt().compute_posterior(y, y_mask, x_states, x_mask)
        #         z_fin = z_fin[:, :, :latent_dim]
        #     else:
        #         z_fin = z_delta
        #
        # z_ini = z_ini.detach().clone()
        # z_ini.requires_grad = True
        with torch.no_grad():
            y_noise, _ = random_token_corruption(y, self._tgt_vocab_size, 0.2)
            z_ini = self.nmt().compute_posterior(y_noise, y_mask, x_states, x_mask)
            stddev = z_ini[:, :, latent_dim:]
            z_ini = z_ini[:, :, :latent_dim] + stddev * torch.randn_like(stddev)
        # z_diff = (z_fin - z_ini).detach() # [batch_size, targets_length, latent_size]
        # with torch.no_grad():
        #     # logits = self.get_logits(z_fin, y_mask, x_states, x_mask)
        #     # y_delta = logits.argmax(-1)
        #     y_noise, _ = random_token_corruption(y, self._tgt_vocab_size, 0.2)
        #     z_noise = self.nmt().compute_posterior(y_noise, y_mask, x_states, x_mask)
        #     z_fin = self.nmt().compute_posterior(y, y_mask, x_states, x_mask)
        #     z_fin = z_fin[:, :, :latent_dim]
        #     stddev = z_noise[:, :, latent_dim:]
        #     z_ini = z_noise[:, :, :latent_dim] + stddev * torch.randn_like(stddev)
        #     z_diff = (z_fin - z_ini).detach()
        # z_ini.requires_grad_(True)

        score = self.score_fn.score(z_ini, y_mask, x_states, x_mask) # [batch_size, targets_length, latent_size]
        updated_z = z_ini + score
        h = self.lat2hid(updated_z)
        h = self.decoder(h, y_mask)
        logits = self.expander_nn(h)
        loss = self.xent_loss(logits, y, y_mask)
        y_pred = logits.argmax(2)
        acc = ((y_pred == y).float() * y_mask).sum() / y_mask.sum()

        score_map = {"loss": loss, "acc": acc, "cosine_sim": -loss}
        return score_map

    def forward(self, x, y, sampling=False):
        x_mask = self.to_float(torch.ne(x, 0))
        y_mask = self.to_float(torch.ne(y, 0))
        score_map = self.compute_loss(x, x_mask, y, y_mask)
        self._mycnt += 1 # pretty hacky I know, sorry haha
        return score_map

    def translate(self, x, n_iter):
        """ Testing codes.
        """
        lanmt = self.nmt()
        x_mask = lanmt.to_float(torch.ne(x, 0))
        x_states = lanmt.embed_layer(x)
        x_states = lanmt.x_encoder(x_states, x_mask)

        # Predict length
        x_lens = x_mask.sum(1)
        delta = lanmt.predict_length(x_states, x_mask)
        import pdb;pdb.set_trace()
        y_lens = delta.long() + x_lens.long()
        # y_lens = x_lens
        y_max_len = torch.max(y_lens.long()).item()
        batch_size = list(x_states.shape)[0]
        y_mask = torch.arange(y_max_len)[None, :].expand(batch_size, y_max_len).cuda()
        y_mask = (y_mask < y_lens[:, None])
        y_mask = y_mask.float()
        # y_mask = x_mask

        # Compute p(z|x)
        p_prob = lanmt.compute_prior(y_mask, x_states, x_mask)
        z = p_prob[..., :lanmt.latent_dim]
        z = self.delta_refine(z, y_mask, x_states, x_mask)
        if OPTS.Twithout_ebm:
            z_ = z
        else:
            z_ = self.energy_sgd(z, y_mask, x_states, x_mask, n_iter=n_iter)

        h = self.lat2hid(z_)
        h = self.decoder(h, y_mask)
        logits = self.expander_nn(h)
        y_pred = logits.argmax(-1)
        y_pred = y_pred * y_mask.long()
        return y_pred

    def nmt(self):
        return self._lanmt[0]

def mean_bt(tensor, mask):
    # Average across the Batch and Time dimension (given a binary mask)
    assert tensor.shape == mask.shape
    return (tensor * mask).sum() / mask.sum()

def mean_t(tensor, mask):
    # Average across the Time dimension (given a binary mask)
    assert tensor.shape == mask.shape
    return (tensor * mask).sum(1) / mask.sum(1)

def cosine_loss_tc(x1, x2, mask):
    # Compute cosine loss over T and C dimension (ignoring PAD)
    # input : [batch_size, targets_length, latent_size]
    # output : [batch_size]
    x1_norm = (x1 ** 2) * mask[:, :, None]
    x1_norm = x1_norm.sum(2).sum(1).sqrt()
    x2_norm = (x2 ** 2) * mask[:, :, None]
    x2_norm = x2_norm.sum(2).sum(1).sqrt()
    num = (x1 * x2) * mask[:, :, None]
    num = num.sum(2).sum(1)
    denom = x1_norm * x2_norm
    sim = num / denom

    return 1 - sim # [bsz]

def cosine_loss_c(x1, x2, mask):
    # Compute cosine loss over C dimension (ignoring PAD)
    # input : [batch_size, targets_length, latent_size]
    # output : [batch_size]
    sim = F.cosine_similarity(x1, x2, dim=2)
    sim = mean_t(sim, mask) # [bsz]
    return 1 - sim

def cosine_loss_t(x1, x2, mask):
    # Compute cosine loss over T dimension (ignoring PAD)
    # input : [batch_size, targets_length]
    # output : [batch_size]
    x1_norm = (x1 ** 2) * mask
    x1_norm = x1_norm.sum(1).sqrt()
    x2_norm = (x2 ** 2) * mask
    x2_norm = x2_norm.sum(1).sqrt()
    num = (x1 * x2) * mask
    num = num.sum(1)
    denom = x1_norm * x2_norm
    sim = num / denom

    return 1 - sim # [bsz]

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
