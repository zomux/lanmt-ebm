#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

def random_token_corruption(seq, vocab_size, ratio=0.2, maskpred=False):
    # seq ~ (Batch, Length)
    if maskpred:
        nosie_tokens = torch.zeros_like(seq).long() + 3
    else:
        nosie_tokens = torch.randint(0, vocab_size, seq.shape)
    mask = (torch.rand(seq.shape) > ratio).float()
    if torch.cuda.is_available():
        nosie_tokens = nosie_tokens.cuda()
        mask = mask.cuda()
    seq = seq.float() * mask + nosie_tokens.float() * (1 - mask)
    seq = seq.long()
    return seq, (1. - mask)
