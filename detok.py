#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

for line in sys.stdin:
    line = line.split("<eoc>")[-1].strip()
    line = line.replace("-LRB-", "(").replace("-RRB-", ")")
    if "▁" in line:
        line = "".join(line.split()).replace("▁", " ").strip()
    tokens = list(line)
    if tokens:
        tokens[0] = tokens[0].upper()
    line = "".join(tokens)
    print(line)

