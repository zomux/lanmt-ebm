#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append(".")

""""
Independent MT refinement model wihtout VAE pre-training.
"""

from lanmt.lib_lanmt_modules import TransformerEncoder
from lanmt.lib_lanmt_model import LANMTModel
from lanmt.lib_simple_encoders import ConvolutionalEncoder
from lanmt.lib_latent_encoder import LatentEncodingNetwork
from lanmt.lib_corrpution import random_token_corruption
from nmtlab.models import Transformer
from nmtlab.utils import OPTS

import random

class IndependentEnergyMT(Transformer):

    def __init__(self, latent_size=256, src_vocab_size=40000, tgt_vocab_size=40000, hidden_size=None):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self._hidden_size = latent_size
        self.set_stepwise_training(False)
        super(IndependentEnergyMT, self).__init__(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
        self.enable_valid_grad = (OPTS.modeltype == "realgrad")

    def prepare(self):
        self.encoder = ConvolutionalEncoder(None, self._latent_size, 3)
        self.decoder = ConvolutionalDecoder(None, self._latent_size, 3)
        self.x_embed = nn.Embedding(self.vocab_size, self._latent_size)
        self.y_embed = nn.Embedding(self.vocab_size, self._latent_size)
        self.expander = nn.Linear(self._latent_size, self.vocab_size)
        # self._encoder = TransformerEncoder(None, self._hidden_size, 3)
        if OPTS.modeltype != "realgrad":
            self.hidden2grad = ConvolutionalEncoder(None, self._latent_size, 3)
        else:
            self.hidden2energy = nn.Sequential(
                nn.Linear(self._hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self._hidden_size // 2, 1)
            )

    def compute_energy(self, z, mask):
        if mask is not None:
            mask = mask.float()
        h = self._latent2hidden(z)
        h = self._encoder(h, mask=mask)
        if OPTS.modeltype != "realgrad":
            grad = self.hidden2grad(h)
            energy = None
        else:
            energy = self._hidden2energy(h)
            mean_energy = ((energy.squeeze(2) * mask).sum(1) / mask.sum(1)).mean()
            grad = torch.autograd.grad(mean_energy, z, create_graph=True)[0]
        return energy, grad

    def compute_loss(self, x, x_mask, y, y_mask):
        bsize, ylen = y.shape
        # Corruption the target sequence to get input
        if OPTS.corruption == "tgt":
            noise_y, noise_mask = random_token_corruption(y, self._tgt_vocab_size)
            noise_y = (noise_y.float() * y_mask).long()
            noise_mask = noise_mask * y_mask
        else:
            raise NotImplementedError
        # Compute encoder
        noise_y_embed = self.y_embed(noise_y)
        noise_z = self.encoder(noise_y_embed, mask=y_mask)
        # Compute energy model and refine the noise_z
        if OPTS.modeltype == "fakegrad":
            refined_z = noise_z
            for _ in range(OPTS.nrefine):
                energy, energy_grad = self.compute_energy(refined_z, y_mask)
                refined_z = refined_z - energy_grad
        elif OPTS.modeltype == "forward":
            _, refined_z = self.compute_energy(noise_z, y_mask)
        # Compute logp for both refined z and noised z
        if OPTS.modeltype == "forward":
            _, refined_z = self.compute_energy(noised_z, mask)
        else:
            refined_z = noised_z
            for _ in range(OPTS.nrefine):
                energy, energy_grad = self.compute_energy(refined_z, mask)
                refined_z = refined_z - energy_grad
        # Compute loss
        logits = self.expander(refined_z)
        # compute cross entropy
        loss_mat = F.cross_entropy(logits.reshape(bsize * ylen, -1), seq.flatten(), reduction="none").reshape(bsize, ylen)
        if OPTS.losstype == "single":
            loss = (loss_mat * mask).sum() / mask.sum()
        elif OPTS.losstype == "balanced":
            loss1 = (loss_mat * mask * (1 - noise_mask)).sum() / (mask * (1 - noise_mask)).sum()
            loss2 = (loss_mat * noise_mask).sum() / noise_mask.sum()
            loss = loss1 + loss2
        else:
            raise NotImplementedError
        # loss = loss2
        yhat = logits.argmax(2)
        acc = ((yhat == seq).float() * mask).sum() / mask.sum()
        noise_acc = ((yhat == seq).float() * noise_mask).sum() / noise_mask.sum()

        return {"loss": loss, "acc": acc, "noise_acc": noise_acc}

    def forward(self, x, y, sampling=False):
        y_mask = self.to_float(torch.ne(y, 0))
        x_mask = self.to_float(torch.ne(x, 0))
        score_map = self.compute_loss(x, x_mask, y, y_mask)
        return score_map

    def refine(self, z, mask=None, n_steps=50, step_size=0.001, return_tokens=False):
        if mask is not None:
            mask = mask.float()
        z = z.clone()
        # step_size = 1. / n_steps
        z.requires_grad_(True)
        for _ in range(n_steps):
            energy, grad = self.compute_energy(z, mask)
            # if not OPTS.evaluate:
            #     print(energy.mean().detach().cpu().numpy(),
            #           grad.norm(2).detach().cpu().numpy())
            # z = z - step_size * grad

            # Denosing updating
            # norm = (grad - z).norm(dim=2)
            # max_pos = norm[:, 2:-1].argmax(1) + 2
            # z[:, max_pos] = grad[:, max_pos]
            # z = grad
            # noise = torch.randn_like(z) * np.sqrt(step_size * 2)
            z = z - step_size * grad
            # norm = grad.norm(dim=2)
            # max_pos = norm.argmax(1)
            # if norm.max() < 0.5:
            #     break
            # z.requires_grad_(False)
            # z[torch.arange(z.shape[0]), max_pos] -= step_size * grad[torch.arange(z.shape[0]), max_pos]
            # z = z.detach()
            # z.requires_grad_(True)
            # print(grad.norm(dim=2))
        # tokens = self.coder().compute_tokens(z, mask)
        if return_tokens:
            tokens = self.expander(z).argmax(2)
            return z, tokens
        else:
            return z, None

