#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
from nmtlab.modules.transformer_modules import MultiHeadAttention
from nmtlab.modules.transformer_modules import TransformerFeedForward
from nmtlab.modules.transformer_modules import residual_connect


class DisentangledEncoder(nn.Module):

    def __init__(self, embed_layer, size, n_layers=3, dropout_ratio=0.1, skip_connect=False):
        super(DisentangledEncoder, self).__init__()
        self.embed_layer = embed_layer
        self.encoder_layers = nn.ModuleList()
        self.skip_connect = skip_connect
        self._rescale = 1. / math.sqrt(2)
        for _ in range(n_layers):
            layer = nn.Sequential(
                nn.LayerNorm(size),
                nn.Linear(size, size),
                nn.ReLU(),
                nn.Dropout(p=dropout_ratio))
            self.encoder_layers.append(layer)

    def forward(self, x, mask=None):
        if self.embed_layer is not None:
            x = self.embed_layer(x)
        first_x = x
        for l, layer in enumerate(self.encoder_layers):
            prev_x = x
            x = layer(x)
            x = prev_x + x
            if self.skip_connect:
                x = self._rescale * (first_x + x)
        return x


class DisentangledCrossEncoder(nn.Module):

    def __init__(self, embed_layer, size, n_layers, n_att_head=8, dropout_ratio=0.1, skip_connect=False):
        super(DisentangledCrossEncoder, self).__init__()
        self._skip = skip_connect
        self._reslace = 1. / math.sqrt(2)
        self.embed_layer = embed_layer
        self.layer_norm = nn.LayerNorm(size)
        self.encoder_layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = DisentangledCrossEncoderLayer(size, dropout_ratio=dropout_ratio)
            self.encoder_layers.append(layer)

    def forward(self, x, x_mask, y, y_mask):
        if self.embed_layer is not None:
            x = self.embed_layer(x)
        first_x = x
        for l, layer in enumerate(self.encoder_layers):
            x = layer(x, x_mask, y, y_mask)
            if self._skip:
                x = self._reslace * (first_x + x)
        x = self.layer_norm(x)
        return x


class DisentangledCrossEncoderLayer(nn.Module):

    def __init__(self, size, n_att_head=8, dropout_ratio=0.1, relative_pos=False):
        super(DisentangledCrossEncoderLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio)
        self.attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio, relative_pos=relative_pos)
        self.ff_layer = nn.Sequential(nn.Linear(size, size), nn.ReLU())
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)

    def forward(self, x, x_mask, y, y_mask):
        h1 = x
        # Attention layer
        h2 = self.layer_norm1(h1)
        h2, _ = self.attention(h2, y, y, mask=y_mask)
        h2 = self.dropout(h2)
        h2 = residual_connect(h2, h1)
        # Feed-forward layer
        h3 = self.layer_norm2(h2)
        h3 = self.ff_layer(h3)
        h3 = self.dropout(h3)
        h3 = residual_connect(h3, h2)
        return h3


class ConvolutionalEncoder(nn.Module):

    def __init__(self, embed_layer, size, n_layers=3, dropout_ratio=0.1, cross_attention=False, skip_connect=False):
        super(ConvolutionalEncoder, self).__init__()
        self.embed_layer = embed_layer
        self.encoder_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.skip_connect = skip_connect
        self._rescale = 1. / math.sqrt(2)
        for _ in range(n_layers):
            self.norm_layers.append(nn.LayerNorm(size))
            layer = nn.Sequential(
                nn.Conv1d(size, size, 3, padding=1),
                nn.ReLU(),
                nn.Dropout(p=dropout_ratio))
            self.encoder_layers.append(layer)

    def forward(self, x, mask=None):
        if self.embed_layer is not None:
            x = self.embed_layer(x)
        first_x = x
        for l, layer in enumerate(self.encoder_layers):
            prev_x = x
            x = self.norm_layers[l](x)
            x = layer(x.transpose(1, 2)).transpose(1, 2)
            if mask is not None:
                x = x * mask[:, :, None]
            x = prev_x + x
            if self.skip_connect:
                x = self._rescale * (first_x + x)
        return x


class TransformerCrossEncoderLayer(nn.Module):

    def __init__(self, size, ff_size=None, n_att_head=8, dropout_ratio=0.1):
        super(TransformerCrossEncoderLayer, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self.dropout = nn.Dropout(dropout_ratio)
        self.conv = nn.Sequential(
                nn.Conv1d(size, size, 3, padding=1),
                nn.ReLU(),
                self.dropout)
        self.cross_attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio)
        self.ff_layer = TransformerFeedForward(size, ff_size, dropout_ratio=dropout_ratio)
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)
        self.layer_norm3 = nn.LayerNorm(size)

    def forward(self, x, x_mask, y, y_mask):
        # Attention layer
        h1 = self.layer_norm1(x)
        h1, _ = self.conv(h1, h1, h1, mask=x_mask)
        h1 = self.dropout(h1)
        h1 = residual_connect(h1, x)
        # Cross-attention
        h2 = self.layer_norm2(h1)
        h2, _ = self.cross_attention(h2, y, y, mask=y_mask)
        h2 = self.dropout(h2)
        h2 = residual_connect(h2, h1)
        # Feed-forward layer
        h3 = self.layer_norm3(h2)
        h3 = self.ff_layer(h3)
        h3 = self.dropout(h3)
        h3 = residual_connect(h3, h2)
        return h