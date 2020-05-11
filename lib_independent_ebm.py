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
from lanmt.lib_simple_encoders import ConvolutionalCrossEncoder
from lanmt.lib_lanmt_modules import TransformerCrossEncoder
from lanmt.lib_latent_encoder import LatentEncodingNetwork
from lanmt.lib_corrpution import random_token_corruption
from nmtlab.models import Transformer
from nmtlab.utils import OPTS

import random

def identity(x, mask=None):
    return x

class IndependentEnergyMT(Transformer):

    def __init__(self, latent_size=256, src_vocab_size=40000, tgt_vocab_size=40000, hidden_size=None):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self._latent_size = latent_size
        self._hidden_size = latent_size
        self.set_stepwise_training(False)
        super(IndependentEnergyMT, self).__init__(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
        self.enable_valid_grad = (OPTS.modeltype == "realgrad" or OPTS.losstype == "scorematch")

    def prepare(self):
        if OPTS.enctype == "identity":
            self.encoder = identity
        elif OPTS.enctype == "conv":
            self.encoder = ConvolutionalEncoder(None, self._latent_size, 3)
        elif OPTS.enctype == "crossconv":
            self.encoder = ConvolutionalCrossEncoder(None, self._latent_size, 3)
        else:
            raise NotImplementedError
        if OPTS.dectype == "identity":
            self.decoder = identity
        elif OPTS.dectype == "conv":
            self.decoder = ConvolutionalEncoder(None, self._latent_size, 3)
        elif OPTS.dectype == "crossconv":
            self.decoder = ConvolutionalCrossEncoder(None, self._latent_size, 3)
        else:
            raise NotImplementedError
        self.x_embed = nn.Embedding(self._src_vocab_size, self._latent_size)
        self.y_embed = nn.Embedding(self._tgt_vocab_size, self._latent_size)
        self.expander = nn.Linear(self._latent_size, self._tgt_vocab_size)
        # self._encoder = TransformerEncoder(None, self._hidden_size, 3)
        if OPTS.ebmtype == "conv":
            self.ebm = ConvolutionalEncoder(None, self._latent_size, 3)
        elif OPTS.ebmtype == "crossconv":
            self.ebm = ConvolutionalCrossEncoder(None, self._latent_size, 3)
        elif OPTS.ebmtype == "crossatt":
            self.ebm = TransformerCrossEncoder(None, self._latent_size, 3)
        else:
            raise NotImplementedError
        self.x_encoder = ConvolutionalEncoder(self.x_embed, self._latent_size, 3)
        if OPTS.modeltype == "realgrad":
            self.latent2energy = nn.Sequential(
                nn.Linear(self._hidden_size, self._hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self._hidden_size // 2, 1)
            )
        if OPTS.mimic:
            self.mimic_ebm = ConvolutionalCrossEncoder(None, self._latent_size, 3)

    def compute_energy(self, z, y_mask, x_states, x_mask):
        if y_mask is not None:
            y_mask = y_mask.float()
        energy = None
        if OPTS.ebmtype.startswith("cross"):
            grad = self.ebm(z, y_mask, x_states, x_mask)
        else:
            grad = self.ebm(z, y_mask)
        if OPTS.modeltype == "realgrad":
            energy = self.latent2energy(grad)
            mean_energy = ((energy.squeeze(2) * y_mask).sum(1) / y_mask.sum(1)).mean()
            grad = torch.autograd.grad(mean_energy, grad, create_graph=True)[0]
        return energy, grad

    def encode(self, x, x_mask, y, y_mask):
        # Pre-compute source states
        if OPTS.ebmtype.startswith("cross"):
            x_states = self.x_encoder(x, x_mask)
        else:
            x_states = None
        # Compute encoder
        noise_y_embed = self.y_embed(y)
        if OPTS.enctype.startswith("cross"):
            noise_z = self.encoder(noise_y_embed, y_mask, x_states, x_mask)
        else:
            noise_z = self.encoder(noise_y_embed, mask=y_mask)
        return noise_z, x_states

    def euc_distance(self, z1, z2, mask):
        distance = ((z2 - z1)**2).sum(2)
        distance = (distance * mask).sum(1) / mask.sum(1)
        return distance.mean()

    def compute_logits(self, x, x_mask, noise_y, y_mask):
        if len(noise_y.shape) == 3:
            refined_z = noise_y
            x_states = x
        else:
            noise_z, x_states = self.encode(x, x_mask, noise_y, y_mask)
            # Compute energy model and refine the noise_z
            if OPTS.modeltype == "fakegrad" or OPTS.modeltype == "realgrad":
                refined_z = noise_z
                for _ in range(OPTS.nrefine):
                    energy, energy_grad = self.compute_energy(refined_z, y_mask, x_states, x_mask)
                    refined_z = refined_z - energy_grad
                if OPTS.mimic and False:
                    mimic_grad = self.mimic_ebm(noise_z.detach(), y_mask, x_states, x_mask)
                    if OPTS.mimic_cos:
                        loss_direction = self.cosine_loss(score, z_diff, y_mask)  # [batch_size]
                    else:
                        OPTS.mimic_loss = self.euc_distance(mimic_grad, energy_grad.detach(), y_mask)

            elif OPTS.modeltype == "forward":
                _, refined_z = self.compute_energy(noise_z, y_mask, x_states, x_mask)
            else:
                raise NotImplementedError
        OPTS.refined_z = refined_z
        if not self.training and OPTS.mimic:
            grad = self.mimic_ebm(refined_z, y_mask, x_states, x_mask)
            refined_z = refined_z + grad * 0.6
            grad = self.mimic_ebm(refined_z, y_mask, x_states, x_mask)
            refined_z = refined_z + grad * 0.6

        # Compute decoder and get logits
        if OPTS.dectype.startswith("cross"):
            decoder_states = self.decoder(refined_z, y_mask, x_states, x_mask)
        else:
            decoder_states = self.decoder(refined_z)
        logits = self.expander(decoder_states)
        return logits

    def xent_loss(self, logits, y, y_mask):
        bsize, ylen = y.shape
        loss_mat = F.cross_entropy(logits.view(bsize * ylen, -1), y.flatten(), reduction="none").view(bsize, ylen)
        loss = (loss_mat * y_mask).sum() / y_mask.sum()
        return loss

    def compute_loss(self, x, x_mask, y, y_mask):
        score_map = {}
        bsize, ylen = y.shape
        # Corruption the target sequence to get input
        if OPTS.corruption == "target":
            noise_y, noise_mask = random_token_corruption(y, self._tgt_vocab_size, OPTS.corrupt, maskpred=OPTS.maskpred)
            noise_y = (noise_y.float() * y_mask).long()
            noise_mask = noise_mask * y_mask
        else:
            raise NotImplementedError
        if OPTS.losstype != "scorematch":
            logits = self.compute_logits(x, x_mask, noise_y, y_mask)
            # compute cross entropy
            loss_mat = F.cross_entropy(logits.view(bsize * ylen, -1), y.flatten(), reduction="none").view(bsize, ylen)
        if OPTS.losstype == "single":
            loss = (loss_mat * y_mask).sum() / y_mask.sum()
            zreg = ((OPTS.refined_z ** 2).sum(2) * y_mask).sum() / y_mask.sum()
            score_map["zabs"] = torch.abs(OPTS.refined_z).mean()
            if OPTS.zreg > 0.0001:
                score_map["zreg"] = zreg
                loss = loss + zreg * OPTS.zreg
            if OPTS.mimic:
                assert OPTS.modeltype == "forward"
                with torch.no_grad():
                    origin_z = OPTS.refined_z
                    refined_y = logits.argmax(2)
                    z, x_states = self.encode(x, x_mask, refined_y, y_mask)
                    _, target_z = self.compute_energy(z, y_mask, x_states, x_mask)
                mimic_grad = self.mimic_ebm(origin_z.detach(), y_mask, x_states, x_mask)
                mimic_loss = self.euc_distance(mimic_grad, (target_z - origin_z).detach(), y_mask)
                score_map["mimic"] = mimic_loss
                loss = mimic_loss
        elif OPTS.losstype == "scorematch":
            noise_z, x_states = self.encode(x, x_mask, noise_y, y_mask)
            target_z, x_states = self.encode(x, x_mask, y, y_mask)
            noise_z = noise_z + torch.randn_like(target_z)
            target_z = target_z + torch.randn_like(target_z)
            # logits = self.compute_logits(x_states, x_mask, noise_z, y_mask)
            # noise_ae_loss = self.xent_loss(logits, noise_y, y_mask)
            noise_ae_loss = 0.
            y_logits = self.compute_logits(x_states, x_mask, target_z, y_mask)
            y_ae_loss = self.xent_loss(y_logits, y, y_mask)
            # Compute score matching loss
            _, grad = self.compute_energy(noise_z, y_mask, x_states, x_mask)
            scorematch_loss = (grad - (target_z - noise_z))**2
            scorematch_loss = scorematch_loss.sum(2)
            scorematch_loss = (scorematch_loss * y_mask).sum() / y_mask.sum()
            loss = scorematch_loss + noise_ae_loss + y_ae_loss
            score_map["zdiff"] = (((target_z - noise_z)**2).sum(2) * y_mask).sum() / y_mask.sum()
            score_map["scorematch"] = scorematch_loss
            # score_map["noise_ae"] = noise_ae_loss
            score_map["y_ae"] = y_ae_loss

        elif OPTS.losstype == "balanced":
            loss1 = (loss_mat * y_mask * (1 - noise_mask)).sum() / (y_mask * (1 - noise_mask)).sum()
            loss2 = (loss_mat * noise_mask).sum() / noise_mask.sum()
            loss = loss1 + loss2
        else:
            raise NotImplementedError
        # loss = loss2
        yhat = logits.argmax(2)
        acc = ((yhat == y).float() * y_mask).sum() / y_mask.sum()
        noise_acc = ((yhat == y).float() * noise_mask).sum() / noise_mask.sum()
        # acc = noise_acc = loss * 0.

        score_map.update({"loss": loss, "acc": acc, "noise_acc": noise_acc})
        return score_map

    def forward(self, x, y, sampling=False):
        y_mask = self.to_float(torch.ne(y, 0))
        x_mask = self.to_float(torch.ne(x, 0))
        score_map = self.compute_loss(x, x_mask, y, y_mask)
        return score_map

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        state_keys = list(state_dict.keys())
        for key in state_keys:
            if key.startswith("module."):
                new_key = key[7:]
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)

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
            z = z + step_size * grad
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

