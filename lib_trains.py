#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab.utils import OPTS, is_root_node

def initialize_trains(project_name):
    tb_logdir = None
    OPTS.trains_task = None
    if is_root_node():
        print("Running on {} GPUs".format(gpu_num))
        if OPTS.tensorboard:
            try:
                from trains import Task
                task = Task.init(project_name="EBM_LM",
                                 task_name=OPTS.result_tag,
                                 auto_connect_arg_parser=False,
                                 output_uri="{}/data/model_backups".format(os.getenv("HOME")))
                task.connect(ap)
                task.set_random_seed(OPTS.seed)
                OPTS.trains_task = task
            except SystemError as e:
                print(e)
                pass
            tb_logdir = os.path.join(OPTS.root, "tensorboard")
            if not os.path.exists(tb_logdir):
                os.mkdir(tb_logdir)