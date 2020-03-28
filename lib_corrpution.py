#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

def random_token_corruption(seq, vocab_size, ratio=0.2):
    # seq ~ (Batch, Length)
    nosie_tokens = torch.randint(0, vocab_size + 1, seq.shape)
    mask = (torch.rand(seq.shape) > ratio).float()
    seq = seq * mask + nosie_tokens * (1 - mask)
    return seq
