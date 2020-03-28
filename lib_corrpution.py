#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def random_token_corruption(seq, vocab_size, ratio=0.2):
    # seq ~ (Batch, Length)
    