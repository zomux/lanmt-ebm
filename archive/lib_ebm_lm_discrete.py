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

from lanmt.lib_lanmt_modules import TransformerEncoder
from lanmt.lib_lanmt_model import LANMTModel
from lanmt.lib_simple_encoders import ConvolutionalEncoder
from lanmt.lib_latent_encoder import LatentEncodingNetwork
from nmtlab.models import Transformer
from nmtlab.utils import OPTS


class DiscreteEnergyLanguageModel(Transformer):

    def __init__(self, coder_model, hidden_size=512, latent_size=None):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self._hidden_size = hidden_size
        self._latent_size = latent_size if latent_size is not None else OPTS.latentdim
        self.set_stepwise_training(False)
        self.compute_real_grad = True
        super(DiscreteEnergyLanguageModel, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        self._coder_model = [coder_model]
        self._coder_model[0].train(False)
        self.enable_valid_grad = False

    def prepare(self):
        # self._encoder = TransformerEncoder(None, self._hidden_size, 3)
        self._encoder = ConvolutionalEncoder(None, self._hidden_size, 3)
        self._latent2hidden = nn.Linear(self._latent_size, self._hidden_size)
        if not self.compute_real_grad:
            self._hidden2latent = nn.Linear(self._hidden_size, self._latent_size)
        else:
            self._hidden2energy = nn.Sequential(
                nn.Linear(self._hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self._hidden_size // 2, 1)
            )

    def compute_energy(self, z, mask):
        if mask is not None:
            mask = mask.float()
        h = self._latent2hidden(z)
        h = self._encoder(h, mask=mask)
        if not self.compute_real_grad:
            grad = self._hidden2latent(h)
            energy = None
        else:
            energy = self._hidden2energy(h)
            mean_energy = ((energy.squeeze(2) * mask).sum(1) / mask.sum(1)).mean()
            grad = torch.autograd.grad(mean_energy, z, create_graph=True)[0]
        return energy, grad

    def compute_loss(self, seq, mask):
        # Compute cross-entropy loss and it's gradient
        with torch.no_grad():
            true_z = self.coder().compute_codes(seq).detach()
        # Compute delta inference
        noise = torch.randn_like(true_z)
        b_mask = (torch.rand(noise.shape[:2]) > 0.2).float()
        if torch.cuda.is_available():
            b_mask = b_mask.cuda()
        noise = noise * b_mask[:, :, None]
        noised_z = true_z + noise
        noised_z.requires_grad_(True)
        # Compute logp for both refined z and noised z
        # with torch.no_grad():
        #     true_logp = self.coder().compute_tokens(true_z, mask, return_logp=True)
        #     noised_logp = self.coder().compute_tokens(noised_z, mask, return_logp=True)
        # Compute energy scores
        energy, energy_grad = self.compute_energy(noised_z, mask)
        # Compute loss
        # score_match_loss = (((energy_grad * (true_z - noised_z) * mask[:, :, None]).sum(2).sum(1) - (true_logp - noised_logp))**2).mean()
        score_match_loss = ((noise - energy_grad)**2).sum(2)
        # score_match_loss = ((true_z - energy_grad)**2).sum(2)
        score_match_loss = ((score_match_loss * mask).sum(1) / mask.sum(1)).mean()
        return {"loss": score_match_loss}

    def forward(self, x, y, sampling=False):
        mask = self.to_float(torch.ne(y, 0))
        score_map = self.compute_loss(y, mask)
        return score_map

    def refine(self, z, mask=None, n_steps=50, step_size=0.001, return_tokens=False):
        if mask is not None:
            mask = mask.float()
        z = z.clone()
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
            z = grad
            # noise = torch.randn_like(z) * np.sqrt(step_size * 2)
            # z = z + step_size * grad + noise
            # norm = grad.norm(dim=2)
            # max_pos = norm.argmax(1)
            # if norm.max() < 0.5:
            #     break
            # z.requires_grad_(False)
            # z[torch.arange(z.shape[0]), max_pos] -= step_size * grad[torch.arange(z.shape[0]), max_pos]
            # z = z.detach()
            # z.requires_grad_(True)
            # print(grad.norm(dim=2))
        tokens = self.coder().compute_tokens(z, mask)
        if return_tokens:
            return tokens
        else:
            return z

    def coder(self):
        coder = self._coder_model[0]
        assert isinstance(coder, LatentEncodingNetwork)
        return coder

if __name__ == '__main__':
    import sys
    sys.path.append(".")
    coder = LatentEncodingNetwork(latent_dim=256, src_vocab_size=1000, tgt_vocab_size=1000)
    # Testing
    lm = EnergyLanguageModel(coder, latent_size=256)
    x = torch.tensor([[1,2,3,4,5]])
    y = torch.tensor([[1,2,3]])
    if torch.cuda.is_available():
        coder.cuda()
        lm.cuda()
        x = x.cuda()
        y = y.cuda()
    lm(x, y)