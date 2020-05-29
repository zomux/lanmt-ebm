#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import json

src_path = "{}/data/wmt14_ende_fair/test.en".format(os.getenv("HOME"))
ref_path = "{}/data/wmt14_ende_fair/test.de".format(os.getenv("HOME"))
src_lens = np.array([len(l.strip().split()) for l in open(src_path)])
ref_lens = np.array([len(l.strip().split()) for l in open(ref_path)])
pred_lens = json.loads(open("/tmp/length_collector.txt").read().strip())
pred_lens = np.array(pred_lens) - 2
print(ref_lens.shape, pred_lens.shape)
diff = (ref_lens - pred_lens) / src_lens
print("mean", diff.mean())
print("std", diff.std())


