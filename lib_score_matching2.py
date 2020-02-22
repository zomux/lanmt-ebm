#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lanmt.lib_lanmt_modules import TransformerEncoder
from lanmt.lib_lanmt_model import LANMTModel
from nmtlab.models import Transformer
from nmtlab.utils import OPTS


class LatentScoreNetwork2(Transformer):

    def __init__(self, lanmt_model, hidden_size=512, latent_size=8):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self.set_stepwise_training(False)
        self.compute_real_grad = False
        super(LatentScoreNetwork2, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        lanmt_model.train(False)
        self._lanmt = [lanmt_model]
        self.enable_valid_grad = True

    def prepare(self):
        self._encoder = TransformerEncoder(None, self._hidden_size, 3)
        self._latent2hidden = nn.Linear(self._latent_size, self._hidden_size)
        if not self.compute_real_grad:
            self._hidden2latent = nn.Linear(self._hidden_size, self._latent_size)
        else:
            self._hidden2energy = nn.Sequential(
                nn.Linear(self._hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self._hidden_size // 2, 1)
            )

    def compute_energy(self, latent, x, mask):
        if mask is not None:
            mask = mask.float()
        h = self._latent2hidden(latent)
        x_embeds = self.nmt().x_embed_layer(x)
        h = self._encoder(h + x_embeds, mask=mask)
        if self.compute_real_grad:
            energy = self._hidden2energy(h)
            mean_energy = ((energy.squeeze(2) * mask).sum(1) / mask.sum(1)).mean()
            grad = torch.autograd.grad(mean_energy, latent, create_graph=True)[0]
        else:
            energy = None
            grad = self._hidden2latent(h)
        return energy, grad

    def compute_token_loss(self, x, x_mask, y, y_mask, latent):
        lanmt = self.nmt()
        prior_states = lanmt.prior_encoder(x, x_mask)
        length_scores = lanmt.compute_length_predictor_loss(prior_states, latent, x_mask, y_mask)
        z_with_y_length = lanmt.convert_length(latent, x_mask, y_mask.sum(-1))
        decoder_states = lanmt.decoder(z_with_y_length, y_mask, prior_states, x_mask)
        logits = lanmt.expander_nn(decoder_states)
        flat_nll = F.cross_entropy(logits.view(-1, logits.shape[2]), y.view(-1), reduction="none")
        nll = flat_nll.view(*y.shape)
        nll = (nll * y_mask).sum(1).mean()
        token_loss = nll + length_scores["len_loss"]
        return token_loss

    def compute_likelihood_gradient(self, x, x_mask, y, y_mask, latent):
        latent_vector = self._latent2hidden(latent)
        token_loss = self.compute_token_loss(x, x_mask, y, y_mask, latent_vector)
        grad = torch.autograd.grad(token_loss, latent)[0]
        return grad

    def compute_loss(self, x, x_mask, y, y_mask):
        # base_latent = x.new_zeros((x.shape[0], x.shape[1], self._latent_size), requires_grad=True, dtype=torch.float)
        # base_latent = torch.randn_like(base_latent) + base_latent
        with torch.no_grad():
            x_embed = self.nmt().x_embed_layer(x)
            q_states = self.nmt().compute_Q_states(x_embed, x_mask, y, y_mask)
            base_latent, _ = self.nmt().bottleneck(q_states, sampling=False)
        base_latent = base_latent + torch.randn_like(base_latent)
        base_latent.requires_grad_(True)
        # Compute cross-entropy loss and it's gradient
        ll_grad = self.compute_likelihood_gradient(x, x_mask, y, y_mask, base_latent)
        # Compute energy scores
        _, energy_grad = self.compute_energy(base_latent, x, x_mask)
        # Compute loss
        score_match_loss = ((ll_grad.detach() - energy_grad)**2).sum(2)
        score_match_loss = ((score_match_loss * x_mask).sum(1) / x_mask.sum(1)).mean()
        return {"loss": score_match_loss * 100.}

    def forward(self, x, y, sampling=False):
        x_mask = self.to_float(torch.ne(x, 0))
        y_mask = self.to_float(torch.ne(y, 0))
        score_map = self.compute_loss(x, x_mask, y, y_mask)
        return score_map

    def refine(self, z, x_states, mask=None, n_steps=50, step_size=10.):
        for _ in range(n_steps):
            _, grad = self.compute_energy(z, x_states, mask)
            if not OPTS.evaluate:
                print(z.norm(2).detach().cpu().numpy(), grad.norm(2).detach().cpu().numpy())
            # z = z - step_size * grad
            # noise = torch.randn_like(z) * np.sqrt(step_size * 2)
            # z = z + step_size * grad + noise
            norm = grad.norm(dim=2)
            max_pos = norm.argmax(1)
            # if norm.max() < 0.5:
            #     break
            z[torch.arange(z.shape[0]), max_pos] -= step_size * grad[torch.arange(z.shape[0]), max_pos]
            # print(grad.norm(dim=2))
        if not OPTS.evaluate:
            raise SystemExit
        return z

    def nmt(self):
        return self._lanmt[0]

if __name__ == '__main__':
    # Testing
    lanmt = LANMTModel(
        src_vocab_size=1000, tgt_vocab_size=1000,
        prior_layers=1, decoder_layers=1)
    snet = LatentScoreNetwork2(lanmt)
    x = torch.tensor([[1,2,3,4,5]])
    y = torch.tensor([[1,2,3]])
    snet(x, y)