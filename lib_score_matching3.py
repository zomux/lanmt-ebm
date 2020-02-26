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
from nmtlab.models import Transformer
from nmtlab.utils import OPTS


class LatentScoreNetwork3(Transformer):

    def __init__(self, lanmt_model, hidden_size=512, latent_size=8):
        """
        Args:
            lanmt_model(LANMTModel)
        """
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self.set_stepwise_training(False)
        self.compute_real_grad = True
        super(LatentScoreNetwork3, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        lanmt_model.train(False)
        self._lanmt = [lanmt_model]
        self.enable_valid_grad = True

    def prepare(self):
        # self._encoder = TransformerEncoder(None, self._hidden_size, 3)
        self._encoder = ConvolutionalEncoder(None, self._hidden_size, 3)
        self._latent2hidden = nn.Linear(self._latent_size, self._hidden_size)
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
        energy = self._hidden2energy(h)
        mean_energy = ((energy.squeeze(2) * mask).sum(1) / mask.sum(1)).mean()
        grad = torch.autograd.grad(mean_energy, latent, create_graph=True)[0]
        return energy, grad

    def compute_logits(self, latent_vec, prior_states, x_mask):
        length_delta = lanmt.predict_length(prior_states, latent_vec, x_mask)
        converted_z, y_mask, y_lens = lanmt.convert_length_with_delta(latent_vec, x_mask, length_delta + 1)
        decoder_states = lanmt.decoder(converted_z, y_mask, prior_states, x_mask)
        logits = lanmt.expander_nn(decoder_states)
        


    def compute_delta_inference(self, x, x_mask, latent, prior_states=None):
        lanmt = self.nmt()
        if prior_states is None:
            prior_states = lanmt.prior_encoder(x, x_mask)
        latent_vec = lanmt.latent2vector_nn(latent)
        length_delta = lanmt.predict_length(prior_states, latent_vec, x_mask)
        converted_z, y_mask, y_lens = lanmt.convert_length_with_delta(latent_vec, x_mask, length_delta + 1)
        decoder_states = lanmt.decoder(converted_z, y_mask, prior_states, x_mask)
        logits = lanmt.expander_nn(decoder_states)
        y = logits.argmax(-1)
        q_states = lanmt.compute_Q_states(lanmt.x_embed_layer(x), x_mask, y, y_mask)
        sampled_z, _ = lanmt.bottleneck(q_states, sampling=False)
        return sampled_z

    def compute_loss(self, x, x_mask):
        # Compute cross-entropy loss and it's gradient
        base_latent = x.new_zeros((x.shape[0], x.shape[1], self._latent_size), requires_grad=True, dtype=torch.float)
        base_latent = torch.randn_like(base_latent) + base_latent
        # Compute delta inference
        with torch.no_grad():
            refined_z = self.compute_delta_inference(x, x_mask, base_latent).detach()
        noise = torch.randn_like(refined_z)
        noised_z = refined_z + noise
        noised_z.requires_grad_(True)
        # Compute energy scores
        energy, energy_grad = self.compute_energy(noised_z, x, x_mask)
        # Compute loss
        score_match_loss = ((noise - energy_grad)**2).sum(2)
        score_match_loss = ((score_match_loss * x_mask).sum(1) / x_mask.sum(1)).mean()
        return {"loss": score_match_loss * 100.}

    def forward(self, x, y, sampling=False):
        x_mask = self.to_float(torch.ne(x, 0))
        score_map = self.compute_loss(x, x_mask)
        return score_map

    def refine(self, z, x, mask=None, n_steps=50, step_size=0.001):
        if mask is not None:
            mask = mask.float()
        if not OPTS.evaluate:
            with torch.no_grad():
                refined_z = self.compute_delta_inference(x, mask, z).detach()
        for _ in range(n_steps):
            energy, grad = self.compute_energy(z, x, mask)
            if not OPTS.evaluate:
                print((z - refined_z).norm(2).detach().cpu().numpy(),
                      energy.mean().detach().cpu().numpy(),
                      grad.norm(2).detach().cpu().numpy())
            z = z - step_size * grad
            # noise = torch.randn_like(z) * np.sqrt(step_size * 2)
            # z = z + step_size * grad + noise
            # norm = grad.norm(dim=2)
            # max_pos = norm.argmax(1)
            # if norm.max() < 0.5:
            #     break
            # z[torch.arange(z.shape[0]), max_pos] += step_size * grad[torch.arange(z.shape[0]), max_pos]
            # print(grad.norm(dim=2))
        if not OPTS.evaluate:
            raise SystemExit
        return z

    def nmt(self):
        return self._lanmt[0]

if __name__ == '__main__':
    import sys
    sys.path.append(".")
    # Testing
    lanmt = LANMTModel(
        src_vocab_size=1000, tgt_vocab_size=1000,
        prior_layers=1, decoder_layers=1)
    snet = LatentScoreNetwork3(lanmt)
    x = torch.tensor([[1,2,3,4,5]])
    y = torch.tensor([[1,2,3]])
    if torch.cuda.is_available():
        lanmt.cuda()
        snet.cuda()
        x = x.cuda()
        y = y.cuda()
    snet(x, y)