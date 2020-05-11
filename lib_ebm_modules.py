#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from lib_lanmt_modules import TransformerEncoder, TransformerCrossEncoder
from lib_simple_encoders import ConvolutionalEncoder, ConvolutionalCrossEncoder

class EnergyFn(nn.Module):
    def __init__(self, D_lat, D_hid, dropout_ratio=0.1, n_layers=4, positive=False):
        super(EnergyFn, self).__init__()
        self.lat2hid = nn.Linear(
            D_lat, D_hid)
        self.transformer = TransformerCrossEncoder(
            embed_layer=None, size=D_hid, n_layers=n_layers, dropout_ratio=dropout_ratio)
        self.hid2energy = nn.Linear(D_hid, 1)
        self.positive = positive

    def forward(self, z, y_mask, x_states, x_mask):
        # input : [batch_size, targets_length, latent_size]
        # output : [batch_size]
        h = self.lat2hid(z)
        energy = self.transformer(h, y_mask, x_states, x_mask)
        energy = energy * y_mask[:, :, None]
        energy = energy.sum(1) / y_mask.sum(1).float()[:, None]
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

# Using ConvNet
class ConvNetEnergyFn(nn.Module):
    def __init__(self, D_lat, D_hid, dropout_ratio=0.1, D_mlp_hid=100, n_layers=3, positive=False):
        super(ConvNetEnergyFn, self).__init__()
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
    def __init__(self, D_lat, D_hid, D_fin, dropout_ratio=0.1, n_layers=4, positive=False):
        super(ScoreFn, self).__init__()
        self.lat2hid = nn.Linear(
            D_lat, D_hid)
        self.transformer = TransformerCrossEncoder(
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

    def score(self, z, y_mask, x_states, x_mask, create_graph=False):
        return self.forward(z, y_mask, x_states, x_mask)

def mean_bt(tensor, mask):
    # Average across the Batch and Time dimension (given a binary mask)
    assert tensor.shape == mask.shape
    return (tensor * mask).sum() / mask.sum()

def mean_t(tensor, mask):
    # Average across the Time dimension (given a binary mask)
    assert tensor.shape == mask.shape
    return (tensor * mask).sum(1) / mask.sum(1)

def score_matching_loss(score, grad, mask):
    # Compute cosine loss over T and C dimension (ignoring PAD)
    # input : [batch_size, targets_length, latent_size]
    # output : [batch_size]
    length = mask.sum(1)
    score_norm_squared = (score ** 2) * mask[:, :, None]
    score_norm_squared = score_norm_squared.sum(2).sum(1)
    dot_prod = (score * grad) * mask[:, :, None]
    dot_prod = dot_prod.sum(2).sum(1)
    #return (score_norm_squared - 2 * dot_prod) / length
    return score_norm_squared - 2 * dot_prod

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
