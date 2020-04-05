#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

def load_horovod():
    horovod_installed = importlib.util.find_spec("horovod") is not None
    if torch.cuda.is_available() and horovod_installed:
        import horovod.torch as hvd
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        part_index = hvd.rank()
        part_num = hvd.size()
        gpu_num = hvd.size()
    else:
        part_index = 0
        part_num = 1
        gpu_num = 1
    

