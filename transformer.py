#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os, sys
from torch import optim
import importlib

import torch
from nmtlab import MTTrainer, MTDataset
from nmtlab.utils import OPTS
from nmtlab.models import Transformer
from nmtlab.schedulers import TransformerScheduler
from nmtlab.utils import is_root_node
sys.path.append(".")
sys.path.append("./nmtlab")

from argparse import ArgumentParser
from datasets import get_dataset_paths

ap = ArgumentParser()
ap.add_argument("--resume", action="store_true")
ap.add_argument("--test", action="store_true")
ap.add_argument("--train", action="store_true")
ap.add_argument("--evaluate", action="store_true")
ap.add_argument("--all", action="store_true")

ap.add_argument("--root", type=str, default="")
ap.add_argument("--opt_batchtokens", type=int, default=8192)
ap.add_argument("--opt_hiddensz", type=int, default=512)
ap.add_argument("--opt_embedsz", type=int, default=512)
ap.add_argument("--opt_heads", type=int, default=8)
ap.add_argument("--opt_encl", type=int, default=6, help="number of encoder layers")
ap.add_argument("--opt_decl", type=int, default=6, help="number of decoder layers")
ap.add_argument("--opt_shard", type=int, default=32)
ap.add_argument("--opt_clipnorm", type=float, default=0)
ap.add_argument("--opt_labelsmooth", type=float, default=0)
ap.add_argument("--opt_criteria", default="loss", type=str)
ap.add_argument("--opt_dtok", default="", type=str)
ap.add_argument("--opt_weightdecay", action="store_true")
ap.add_argument("--opt_marginloss", action="store_true")
ap.add_argument("--opt_warmsteps", type=int, default=4000)
ap.add_argument("--opt_maxsteps", type=int, default=100000)
ap.add_argument("--opt_seed", type=int, default=3)
ap.add_argument("--opt_nohvd", action="store_true", help="get rid of horovod")
ap.add_argument("--opt_chkavg", action="store_true")
ap.add_argument("--opt_Tbeam", type=int, default=3)
ap.add_argument("--opt_Twmt17", action="store_true")
ap.add_argument("--model_path",
                default="DATA_ROOT/models/transformer.pt")
ap.add_argument("--result_path",
                default="DATA_ROOT/results/transformer.result")
OPTS.parse(ap)

OPTS.model_path = OPTS.model_path.replace("DATA_ROOT", OPTS.root)
OPTS.result_path = OPTS.result_path.replace("DATA_ROOT", OPTS.root)

# Determine the number of GPUs to use
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

# Get the path variables
(
    train_src_corpus,
    train_tgt_corpus,
    distilled_tgt_corpus,
    truncate_datapoints,
    test_src_corpus,
    test_tgt_corpus,
    ref_path,
    src_vocab_path,
    tgt_vocab_path,
    n_valid_per_epoch,
    training_warmsteps,
    training_maxsteps,
    pretrained_autoregressive_path
) = get_dataset_paths(OPTS.root, OPTS.dtok)

dataset = MTDataset(
        src_corpus=train_src_corpus, tgt_corpus=train_tgt_corpus,
        src_vocab=src_vocab_path, tgt_vocab=tgt_vocab_path,
        batch_size=OPTS.batchtokens * gpu_num, batch_type="token",
        truncate=None, max_length=60,
        n_valid_samples=500)

nmt = Transformer(
    num_encoders=OPTS.encl, num_decoders=OPTS.decl,
    dataset=dataset, hidden_size=OPTS.hiddensz, embed_size=OPTS.embedsz,
    label_uncertainty=OPTS.labelsmooth, n_att_heads=OPTS.heads, shard_size=OPTS.shard
)

if OPTS.train or OPTS.all:
    scheduler = TransformerScheduler(warm_steps=OPTS.warmsteps, max_steps=OPTS.maxsteps)
    weight_decay = 1e-5 if OPTS.weightdecay else 0
    optimizer = optim.Adam(nmt.parameters(), lr=0.0001, weight_decay=weight_decay, betas=(0.9, 0.98))
    trainer = MTTrainer(
        nmt, dataset, optimizer, scheduler=scheduler,
        multigpu=gpu_num > 1, using_horovod=not OPTS.nohvd
    )
    OPTS.trainer = trainer
    chk_average_number = 5 if OPTS.chkavg else 0
    trainer.configure(
        save_path=OPTS.model_path,
        n_valid_per_epoch=n_valid_per_epoch,
        criteria=OPTS.criteria,
        clip_norm=OPTS.clipnorm,
        checkpoint_average=chk_average_number
    )
    if OPTS.resume:
        trainer.load()
    trainer.run()
if OPTS.test or OPTS.all:
    import torch
    from nmtlab.decoding import BeamTranslator
    translator = BeamTranslator(nmt, dataset.src_vocab(), dataset.tgt_vocab(), beam_size=OPTS.Tbeam)
    if OPTS.chkavg:
        chk_count = 0
        state_dict = None
        for i in range(1000):
            path = OPTS.model_path + ".chk{}".format(i)
            if os.path.exists(path):
                chkpoint = torch.load(path)["model_state"]
                if state_dict is None:
                    state_dict = chkpoint
                else:
                    for key, val in chkpoint.items():
                        state_dict[key] += val
                chk_count += 1
        for key in state_dict.keys():
            state_dict[key] /= float(chk_count)
        if is_root_node():
            print("Averaged {} checkpoints".format(chk_count))
        translator.model.load_state_dict(state_dict)
    else:
        assert os.path.exists(OPTS.model_path)
        translator.load(OPTS.model_path)
    translator.batch_translate(test_src_corpus, OPTS.result_path)
    if is_root_node():
        print("[Translation Result]")
        print(OPTS.result_path)
if OPTS.evaluate or OPTS.all:
    from nmtlab.evaluation import MosesBLEUEvaluator, SacreBLEUEvaluator
    if is_root_node():
        print("[tokenized BLEU]")
        with open("/tmp/result.txt", "w") as outf:
            for line in open(OPTS.result_path):
                if not line.strip():
                    result = "x"
                else:
                    result = line.strip()
                outf.write(result + "\n")
        if OPTS.dtok == "wmt14_ende":
            evaluator = SacreBLEUEvaluator(ref_path=ref_path, tokenizer="intl", lowercase=True)
        elif OPTS.dtok.endswith("en"):
            evaluator = SacreBLEUEvaluator(ref_path=ref_path, tokenizer="intl", lowercase=True)
        else:
            evaluator = MosesBLEUEvaluator(ref_path)
        print(OPTS.result_path)
        print(evaluator.evaluate("/tmp/result.txt"))

