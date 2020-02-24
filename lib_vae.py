#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from nmtlab.utils import OPTS


class VAEBottleneck(nn.Module):

    def __init__(self, hidden_size, z_size=None, standard_var=False):
        super(VAEBottleneck, self).__init__()
        self.hidden_size = hidden_size
        self.standard_var = standard_var
        if z_size is None:
            self.z_size = self.hidden_size
        else:
            self.z_size = z_size
        self.dense = nn.Linear(hidden_size, self.z_size * 2)

    def forward(self, x, sampling=True, residual_q=None):
        vec = self.dense(x)
        mu = vec[:, :, :self.z_size]
        if self.standard_var:
            var = vec[:, :, self.z_size:] * 0. + 0.55
        else:
            var = vec[:, :, self.z_size:]
        if residual_q is not None:
            mu = 0.5 * (mu + residual_q[:, :, :self.z_size])
        if not sampling:
            return mu, vec
        else:
            var = F.softplus(var)
            if residual_q is not None:
                var = 0.5 * (var + F.softplus(residual_q[:, :, self.z_size:]))
            noise = mu.clone()
            noise = noise.normal_()
            z = mu + noise * var
            return z, vec

    def sample_any_dist(self, dist, deterministic=False, samples=1, noise_level=1.):
        mu = dist[:, :, :self.z_size]
        if deterministic:
            return mu
        else:
            var = F.softplus(dist[:, :, self.z_size:])
            noise = mu.clone()
            if samples > 1:
                if noise.shape[0] == 1:
                    noise = noise.expand(samples, -1, -1).clone()
                    mu = mu.expand(samples, -1, -1).clone()
                    var = var.expand(samples, -1, -1).clone()
                else:
                    noise = noise[:, None, :, :].expand(-1, samples, -1, -1).clone()
                    mu = mu[:, None, :, :].expand(-1, samples, -1, -1).clone()
                    var = var[:, None, :, :].expand(-1, samples, -1, -1).clone()

            noise = noise.normal_()
            z = mu + noise * var * noise_level
            return z