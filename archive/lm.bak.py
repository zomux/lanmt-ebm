#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This model unifies the training of decoder, latent encoder, latent predictor
"""

from __future__ import division
from __future__ import print_function

import os, sys
import time
import importlib
import torch
from torch import optim
sys.path.append(".")

import nmtlab
from nmtlab import MTTrainer, MTDataset
from nmtlab.utils import OPTS, Vocab
from nmtlab.schedulers import TransformerScheduler, SimpleScheduler
from nmtlab.utils import is_root_node
from nmtlab.utils.monitor import trains_stop_stdout_monitor, trains_restore_stdout_monitor
from argparse import ArgumentParser

from lanmt.lib_latent_encoder import LatentEncodingNetwork
from lib_ebm_lm import EnergyLanguageModel
from datasets import get_dataset_paths

DATA_ROOT = "./mydata"
PRETRAINED_MODEL_MAP = {
    "wmt14_ende": "{}/shu_trained_wmt14_ende.pt".format(DATA_ROOT),
    "aspec_jaen": "{}/shu_trained_aspec_jaen.pt".format(DATA_ROOT),
}
TRAINING_MAX_TOKENS = 60

ap = ArgumentParser()
ap.add_argument("--root", type=str, default=DATA_ROOT)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--test", action="store_true")
ap.add_argument("--train", action="store_true")
ap.add_argument("--evaluate", action="store_true")
ap.add_argument("-tb", "--tensorboard", action="store_true")
ap.add_argument("--opt_dtok", default="", type=str, help="dataset token")
ap.add_argument("--opt_seed", type=int, default=3, help="random seed")

# Commmon option for both autoregressive and non-autoregressive models
ap.add_argument("--opt_batchtokens", type=int, default=4096)
ap.add_argument("--opt_hiddensz", type=int, default=512)
ap.add_argument("--opt_embedsz", type=int, default=512)
ap.add_argument("--opt_heads", type=int, default=8)
ap.add_argument("--opt_longertrain", action="store_true")
ap.add_argument("--opt_x3longertrain", action="store_true")
ap.add_argument("--opt_disentangle", action="store_true")

# Options for LANMT
ap.add_argument("--opt_latentdim", default=256, type=int, help="dimension of latent variables")
ap.add_argument("--opt_distill", action="store_true", help="train with knowledge distillation")


# Paths
ap.add_argument("--model_path",
                default="{}/lm.pt".format(DATA_ROOT))
ap.add_argument("--result_path",
                default="{}/lm.result".format(DATA_ROOT))
OPTS.parse(ap)

OPTS.fix_bug1 = True
OPTS.fix_bug2 = False
OPTS.model_path = OPTS.model_path.replace(DATA_ROOT, OPTS.root)
OPTS.result_path = OPTS.result_path.replace(DATA_ROOT, OPTS.root)

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

# Tensorboard Logging
tb_logdir = None
OPTS.trains_task = None
if is_root_node():
    print("Running on {} GPUs".format(gpu_num))
    if OPTS.tensorboard:
        try:
            from trains import Task
            task = Task.init(project_name="EBM_LM", task_name=OPTS.result_tag, auto_connect_arg_parser=False)
            task.connect(ap)
            task.set_random_seed(OPTS.seed)
            OPTS.trains_task = task
        except SystemError as e:
            print(e)
            pass
        tb_logdir = os.path.join(OPTS.root, "tensorboard")
        if not os.path.exists(tb_logdir):
            os.mkdir(tb_logdir)

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

if OPTS.longertrain:
    training_maxsteps = int(training_maxsteps * 1.5)
if OPTS.x3longertrain:
    training_maxsteps = int(training_maxsteps * 3)

if nmtlab.__version__ < "0.7.0":
    print("lanmt now requires nmtlab >= 0.7.0")
    print("Update by pip install -U nmtlab")
    sys.exit()

# Define dataset
if OPTS.distill:
    tgt_corpus = distilled_tgt_corpus
else:
    tgt_corpus = train_tgt_corpus


n_valid_samples = 5000 if OPTS.finetune else 500
if OPTS.train:
    dataset = MTDataset(
        src_corpus=train_src_corpus, tgt_corpus=tgt_corpus,
        src_vocab=src_vocab_path, tgt_vocab=tgt_vocab_path,
        batch_size=OPTS.batchtokens * gpu_num, batch_type="token",
        truncate=truncate_datapoints, max_length=TRAINING_MAX_TOKENS,
        n_valid_samples=n_valid_samples)
else:
    dataset = None

# Create the model
basic_options = dict(
    dataset=dataset,
    src_vocab_size=Vocab(src_vocab_path).size(),
    tgt_vocab_size=Vocab(tgt_vocab_path).size(),
    hidden_size=OPTS.hiddensz, embed_size=OPTS.embedsz,
    n_att_heads=OPTS.heads, shard_size=OPTS.shard,
    seed=OPTS.seed
)

# lanmt_options = basic_options.copy()
# lanmt_options.update(dict(
#     latent_dim=OPTS.latentdim,
#     KL_budget=0. if OPTS.finetune else OPTS.klbudget,
#     max_train_steps=training_maxsteps,
# ))
#
# vae = LatentEncodingNetwork(**lanmt_options)
# vae_path = "{}/data/wmt14_ende_fair/lacoder_batchtokens-8192_distill_dtok-wmt14_fair_ende_klbudget-15.0_latentdim-{}.pt".format(os.getenv("HOME"), OPTS.latentdim)
# if OPTS.disentangle:
#     vae_path = vae_path.replace("_distill", "_disentangle_distill")
# print("loading", vae_path)
# assert os.path.exists(vae_path)
# vae.load(vae_path)
# if torch.cuda.is_available():
#     vae.cuda()

nmt = EnergyLanguageModel(latent_size=OPTS.latentdim)


# Training
if OPTS.train or OPTS.all:
    # Training code
    scheduler = SimpleScheduler(max_epoch=20)
    # scheduler = TransformerScheduler(warm_steps=training_warmsteps, max_steps=training_maxsteps)
    optimizer = optim.Adam(nmt.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-4)

    trainer = MTTrainer(
        nmt, dataset, optimizer,
        scheduler=scheduler, multigpu=gpu_num > 1,
        using_horovod=horovod_installed)
    OPTS.trainer = trainer
    trainer.configure(
        save_path=OPTS.model_path,
        n_valid_per_epoch=n_valid_per_epoch,
        criteria="loss",
        tensorboard_logdir=tb_logdir,
        # clip_norm=0.1 if OPTS.scorenet else 0
    )
    if OPTS.resume:
        trainer.load()
    trains_stop_stdout_monitor()
    trainer.run()
    trains_restore_stdout_monitor()

# Translation
if OPTS.test or OPTS.all:
    # Translate using only one GPU
    if not is_root_node():
        sys.exit()
    torch.manual_seed(OPTS.seed)
    model_path = OPTS.model_path
    if not os.path.exists(model_path):
        print("Cannot find model in {}".format(model_path))
        sys.exit()
    nmt.load(model_path)
    if torch.cuda.is_available():
        nmt.cuda()
    nmt.train(False)
    src_vocab = Vocab(src_vocab_path)
    tgt_vocab = Vocab(tgt_vocab_path)

    # Testing for langauge model
    lines = open(test_tgt_corpus).readlines()
    first_line = lines[0]
    first_line = "Gut@@ ach : Noch ach Sicherheit ach Fußgän@@ ger ."
    # first_line = "ach ach ."
    print(first_line)
    first_line_tokens = tgt_vocab.encode("<s> {} </s>".format(first_line.strip()).split())
    input = torch.tensor([first_line_tokens])
    if torch.cuda.is_available():
        input = input.cuda()
    # z = vae.compute_codes(input)
    z = nmt.compute_prior_states(input)
    # z = torch.zeros((1, 6, OPTS.latentdim))
    mask = torch.ones((1, z.shape[1]))
    if torch.cuda.is_available():
        mask = mask.cuda()
        z = z.cuda()
    init_z = z.clone()
    for _ in range(10):
        z, tokens = nmt.refine(z, mask, n_steps=1, step_size=0.5, return_tokens=True)
        # z[:, 0] = init_z[:, 0]
        # z[:, -1] = init_z[:, -1]
        line = tgt_vocab.decode(tokens[0])
        print(" ".join(line))
    raise SystemExit


    result_path = OPTS.result_path
    # Read data
    lines = open(test_tgt_corpus).readlines()
    trains_stop_stdout_monitor()
    with open(OPTS.result_path, "w") as outf:
        for i, line in enumerate(lines):
            # Make a batch
            tokens = tgt_vocab.encode("<s> {} </s>".format(line.strip()).split())
            x = torch.tensor([tokens])
            if torch.cuda.is_available():
                x = x.cuda()
            mask = torch.ne(x, 0).float()
            # Compute codes
            codes = nmt.compute_codes(x)
            tokens = nmt.compute_tokens(codes, mask)
            # Predict latent and target words from prior
            target_tokens = tokens.cpu().numpy()[0].tolist()
            # Convert token IDs back to words
            target_tokens = [t for t in target_tokens if t > 2]
            target_words = tgt_vocab.decode(target_tokens)
            target_sent = " ".join(target_words)
            import pdb;pdb.set_trace()
            outf.write(target_sent + "\n")
            sys.stdout.write("\rtranslating: {:.1f}%  ".format(float(i) * 100 / len(lines)))
            sys.stdout.flush()
    sys.stdout.write("\n")
    trains_restore_stdout_monitor()


