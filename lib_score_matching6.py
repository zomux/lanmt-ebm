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
from nmtlab.models import Transformer
from nmtlab.utils import OPTS

from lib_envswitch import envswitch

class LatentScoreNetwork6(Transformer):

    def __init__(
        self, lanmt_model, hidden_size=256, latent_size=8,
        noise=0.1, targets="logpy", decoder="fixed", imitation=False,
        imit_rand_steps=1, cosine="T", enable_valid_grad=True):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self.imitation = imitation
        self.imit_rand_steps = imit_rand_steps
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self.set_stepwise_training(False)
        super(LatentScoreNetwork6, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        lanmt_model.train(False)
        self._lanmt = [lanmt_model]
        self.enable_valid_grad = True
        self.train()
        self._mycnt = 0
        self.cosine_loss = cosine_loss_c if cosine=="C" else cosine_loss_tc

        self.noise = noise

        if envswitch.who() == "shu":
            main_dir = "{}/data/wmt14_ende_fair/tensorboard".format(os.getenv("HOME"))
        else:
            main_dir = "/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/tensorboard/"

    def prepare(self):
        self.energy_lat2hid = nn.Linear(self._latent_size, self._hidden_size)
        self.energy_transformer = TransformerCrossEncoder(
            embed_layer=None, size=self._hidden_size, n_layers=3, dropout_ratio=0.1)
        self.energy_conv = ConvolutionalEncoder(
            embed_layer=None, size=self._hidden_size, n_layers=3, dropout_ratio=0.1,
            cross_attention=False, skip_connect=False)
        self.energy_linear = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self._hidden_size, 100),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(100, 1)
        )

        self.magnitude_lat2hid = nn.Linear(self._latent_size, self._hidden_size)
        self.magnitude_transformer = TransformerCrossEncoder(
            embed_layer=None, size=self._hidden_size, n_layers=3, dropout_ratio=0.1)
        self.magnitude_conv = ConvolutionalEncoder(
            embed_layer=None, size=self._hidden_size, n_layers=3, dropout_ratio=0.1,
            cross_attention=False, skip_connect=False)
        self.magnitude_linear = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self._hidden_size, 100),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(100, 1)
        )

    def compute_energy(self, z, y_mask, x_states, x_mask):
        # z : [bsz, y_length, lat_size]
        h = self.energy_lat2hid(z)  # [bsz, y_length, hid_size]
        energy = self.energy_transformer(h, y_mask, x_states, x_mask)  # [bsz, y_length, hid_size]
        energy_states = energy * y_mask[:, :, None]
        energy_states = self.energy_conv(energy_states, y_mask)
        energy_states = energy_states.max(dim=1)[0]
        energy = self.energy_linear(energy_states)  # [bsz, 1]
        return energy[:, 0]

    def compute_magnitude(self, z, y_mask, x_states, x_mask):
        # z : [bsz, y_length, lat_size]
        h = self.magnitude_lat2hid(z)  # [bsz, y_length, hid_size]
        magnitude = self.magnitude_transformer(h, y_mask, x_states, x_mask)  # [bsz, y_length, hid_size]
        magnitude_states = magnitude * y_mask[:, :, None]
        magnitude_states = self.magnitude_conv(magnitude_states, y_mask)
        magnitude_states = magnitude_states.max(dim=1)[0]
        magnitude = self.magnitude_linear(magnitude_states)  # [bsz, 1]
        return magnitude[:, 0]

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

    def energy_sgd(self, z, y_mask, x_states, x_mask, n_iter):
        # z : [bsz, y_length, lat_size]
        y_length = y_mask.sum(1).float()
        scores = []
        lrs = [0.3, 0.1]
        for idx in range(n_iter):
            z = z.detach().clone()
            z.requires_grad = True
            magnitude = self.compute_magnitude(z, y_mask, x_states, x_mask) # [bsz]

            energy = self.compute_energy(z, y_mask, x_states, x_mask)
            dummy = torch.ones_like(energy)
            dummy.requires_grad = True
            grad = autograd.grad(energy, z, create_graph=False, grad_outputs=dummy)
            score = grad[0].detach() # [bsz, length, latent_dim]
            scores.append(score)
            score_norm = (score ** 2) * y_mask[:, :, None]
            score_norm = score_norm.sum(2).sum(1).sqrt()
            multiplier = magnitude * y_length.sqrt() / score_norm
            score = score * multiplier[:, None, None] * lrs[idx]

            z = z + score
        return z, scores

    def energy_line_search(self, z, y_mask, x_states, x_mask, p_prob, n_iter, c=0.05, tau=0.5):
        # implementation from https://en.wikipedia.org/wiki/Backtracking_line_search
        # hyperparameters : alpha, c, tau taken from the wiki page.
        # z : [bsz, y_length, lat_size]
        scores = []
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
            if idx == 0:
                scores.append(score)

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
        return z, scores

    def get_logits(self, z, y_mask, x_states, x_mask):
        lanmt = self.nmt()
        hid = lanmt.lat2hid(z)
        decoder_states = lanmt.decoder(hid, y_mask, x_states, x_mask).detach()
        logits = lanmt.expander_nn(decoder_states)
        return logits

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
            if self.noise == "rand":
                stddev = stddev * np.random.random_sample()
            z_ini += stddev

        z_ini = z_ini.detach().clone()
        z_ini.requires_grad = True
        z_fin = self.delta_refine(z_ini, y_mask, x_states, x_mask)
        z_diff = (z_fin - z_ini).detach() # [bsz, targets_length, lat_size]

        energy = self.compute_energy(z_ini, y_mask, x_states, x_mask) # [bsz]
        dummy = torch.ones_like(energy)
        dummy.requires_grad = True
        grad = autograd.grad(energy, z_ini, create_graph=True, grad_outputs=dummy)
        score = grad[0] # [bsz, targets_length, lat_size]
        loss_direction = self.cosine_loss(score, z_diff, y_mask) # [bsz]

        magnitude = self.compute_magnitude(z_ini, y_mask, x_states, x_mask) # [bsz]
        z_diff_norm = (z_diff ** 2) * y_mask[:, :, None]
        z_diff_norm = z_diff_norm.sum(2).sum(1).sqrt() / y_length.sqrt() # euclidean norm normalized over length
        loss_magnitude = (magnitude - z_diff_norm) ** 2

        loss = loss_magnitude + loss_direction
        loss = loss.mean(0)
        score_map = {"loss": loss}
        energy, dummy, score, grad = None, None, None, None

        if not self.training or self._mycnt % 50 == 0:
            score_map["cosine_sim"] = 1 - loss_direction.mean()
            score_map["z_ini_norm"] = mean_bt(z_ini.norm(dim=2), y_mask)
            score_map["z_fin_norm"] = mean_bt(z_fin.norm(dim=2), y_mask)
            score_map["z_diff_norm"] = mean_bt(z_diff.norm(dim=2), y_mask)

            score_map["loss_magnitude"] = loss_magnitude.mean()
            score_map["loss_direction"] = loss_direction.mean()

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
        if OPTS.Twithout_ebm:
            z_ = z
        else:
            z_, _ = self.energy_sgd(z, y_mask, x_states, x_mask, n_iter=n_iter)

        logits = self.get_logits(z_, y_mask, x_states, x_mask)
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
    x1_norm = (x1 ** 2) ** mask[:, :, None]
    x1_norm = x1_norm.sum(2).sum(1).sqrt()
    x2_norm = (x2 ** 2) ** mask[:, :, None]
    x2_norm = x2_norm.sum(2).sum(1).sqrt()
    num = (x1 * x2) * mask[:, :, None]
    num = num.sum(2).sum(1)
    denom = x1_norm * x2_norm
    sim = num / denom

    return 1 - sim # [bsz]

def cosine_loss_c(x1, x2, mask):
    # Average across the Time dimension (given a binary mask)
    sim = F.cosine_similarity(x1, x2, dim=2)
    sim = mean_t(sim, mask) # [bsz]
    return 1 - sim

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
