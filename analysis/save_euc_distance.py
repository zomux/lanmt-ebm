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
import torch.nn.functional as F
sys.path.append(".")

import nmtlab
from nmtlab import MTTrainer, MTDataset
from nmtlab.utils import OPTS, Vocab
from nmtlab.schedulers import TransformerScheduler, SimpleScheduler
from nmtlab.utils import is_root_node
from nmtlab.utils.monitor import trains_stop_stdout_monitor, trains_restore_stdout_monitor
from nmtlab.evaluation import MosesBLEUEvaluator, SacreBLEUEvaluator
from collections import defaultdict
import numpy as np
from argparse import ArgumentParser
from contextlib import suppress

from lib_lanmt_model2 import LANMTModel2
from lib_rescoring import load_rescoring_transformer
from lib_envswitch import envswitch
from datasets import get_dataset_paths

DATA_ROOT = "/misc/vlgscratch4/ChoGroup/jason/corpora/iwslt/iwslt16_ende"
TRAINING_MAX_TOKENS = 60

# Shu paths
envswitch.register("shu", "data_root", "{}/data/wmt14_ende_fair".format(os.getenv("HOME")))
envswitch.register("jason_prince", "data_root", "/scratch/yl1363/corpora/iwslt/iwslt16_ende")
#envswitch.register("jason_prince", "data_root", "/scratch/yl1363/corpora/wmt/wmt16/en_ro")
envswitch.register("jason", "data_root", "/misc/vlgscratch4/ChoGroup/jason/corpora/iwslt/iwslt16_ende")

envswitch.register("jason_prince", "home_dir", "/scratch/yl1363/lanmt-ebm")
envswitch.register("jason", "home_dir", "/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm")

DATA_ROOT = envswitch.load("data_root", default=DATA_ROOT)
HOME_DIR = envswitch.load("home_dir", default=DATA_ROOT)
envswitch.register(
    "shu", "lanmt_path",
    os.path.join(DATA_ROOT,
        "lanmt_annealbudget_batchtokens-8192_distill_dtok-wmt14_fair_ende_fastanneal_longertrain.pt"
    )
)
envswitch.register(
    "jason", "lanmt_path", "/misc/vlgscratch4/ChoGroup/jason/lanmt/checkpoints/lanmt_annealbudget_batchtokens-4092_distill_dtok-iwslt16_deen_tied.pt"
)
envswitch.register(
    "jason_prince", "lanmt_path", "/scratch/yl1363/lanmt-ebm/checkpoints_lanmt/lanmt_annealbudget_batchtokens-4092_distill_dtok-iwslt16_deen_tied.pt"
    #"jason_prince", "lanmt_path", "/scratch/yl1363/lanmt-ebm/checkpoints/lvm/iwslt16_deen/lanmt_annealbudget_batchtokens-4092_distill_fastanneal_fixbug2_latentdim-2_lr-0.0003.pt"
    #"jason_prince", "lanmt_path", "/scratch/yl1363/lanmt-ebm/checkpoints/lvm/wmt16_roen/lanmt_annealbudget_batchtokens-8192_distill_dtok-wmt16_roen_fastanneal_longertrain.pt"
)

ap = ArgumentParser()
ap.add_argument("--root", type=str, default=DATA_ROOT)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--test", action="store_true")
ap.add_argument("--batch_test", action="store_true")
ap.add_argument("--train", action="store_true")
ap.add_argument("--evaluate", action="store_true")
ap.add_argument("--analyze_latents", action="store_true")
ap.add_argument("--profile", action="store_true")
ap.add_argument("--test_fix_length", type=int, default=0)
ap.add_argument("--all", action="store_true")
ap.add_argument("-tb", "--tensorboard", action="store_true")
ap.add_argument("--use_pretrain", action="store_true", help="use pretrained model trained by Raphael Shu")
ap.add_argument("--opt_dtok", default="iwslt16_deen", type=str, help="dataset token")
ap.add_argument("--opt_seed", type=int, default=3, help="random seed")
ap.add_argument("--opt_tied", action="store_true")

# Commmon option for both autoregressive and non-autoregressive models
ap.add_argument("--opt_batchtokens", type=int, default=4096)
ap.add_argument("--opt_hiddensz", type=int, default=256)
ap.add_argument("--opt_embedsz", type=int, default=256)
ap.add_argument("--opt_heads", type=int, default=4)
ap.add_argument("--opt_shard", type=int, default=32)
ap.add_argument("--opt_longertrain", action="store_true")
ap.add_argument("--opt_x3longertrain", action="store_true")

# Options for LANMT
ap.add_argument("--opt_priorl", type=int, default=3, help="layers for each z encoder")
ap.add_argument("--opt_decoderl", type=int, default=3, help="number of decoder layers")
ap.add_argument("--opt_latentdim", default=8, type=int, help="dimension of latent variables")

# Options for EBM
ap.add_argument("--opt_ebm_lr", default=0.001, type=float)
ap.add_argument("--opt_ebm_useconv", action="store_true")
ap.add_argument("--opt_direction_n_layers", default=4, type=int)
ap.add_argument("--opt_magnitude_n_layers", default=4, type=int)
ap.add_argument("--opt_noise", default=1.0, type=float)
ap.add_argument("--opt_train_sgd_steps", default=0, type=int)
ap.add_argument("--opt_train_step_size", default=0.0, type=float)
ap.add_argument("--opt_train_delta_steps", default=1, type=int)
ap.add_argument("--opt_clipnorm", action="store_true", help="clip the gradient norm")
ap.add_argument("--opt_modeltype", default="whichgrad", type=str)
ap.add_argument("--opt_ebmtype", default="transformer", type=str)
ap.add_argument("--opt_modelclass", default="", type=str)
ap.add_argument("--opt_corrupt", action="store_true")
ap.add_argument("--opt_Tsgd_steps", default=1, type=int)
ap.add_argument("--opt_Tstep_size", default=0.8, type=float, help="step size for EBM SGD")
ap.add_argument("--opt_Treport_log_joint", action="store_true")
ap.add_argument("--opt_Treport_elbo", action="store_true")
ap.add_argument("--opt_decgrad", action="store_true", help="use decoder gradient as target of score matching")

# Decoding options
ap.add_argument("--opt_Twithout_ebm", action="store_true", help="without using EBM")
ap.add_argument("--opt_distill", action="store_true", help="train with knowledge distillation")
ap.add_argument("--opt_annealbudget", action="store_true", help="switch of annealing KL budget")
ap.add_argument("--opt_fixbug1", action="store_true", help="fix bug in length converter")
ap.add_argument("--opt_fixbug2", action="store_true", help="fix bug in transformer decoder")
ap.add_argument("--opt_scorenet", action="store_true")
ap.add_argument("--opt_denoise", action="store_true")
ap.add_argument("--opt_finetune", action="store_true",
                help="finetune the model without limiting KL with a budget")

# Options only for inference
ap.add_argument("--opt_Trefine_steps", type=int, default=0, help="steps of running iterative refinement")
ap.add_argument("--opt_Tlatent_search", action="store_true", help="whether to search over multiple latents")
ap.add_argument("--opt_Tteacher_rescore", action="store_true", help="whether to use teacher rescoring")
ap.add_argument("--opt_Tcandidate_num", default=50, type=int, help="number of latent candidate for latent search")
ap.add_argument("--opt_Tbatch_size", default=8000, type=int, help="batch size for batch translate")

# Experimental options
ap.add_argument("--opt_fp16", action="store_true")
ap.add_argument("--opt_nokl", action="store_true")
ap.add_argument("--opt_klbudget", type=float, default=1.0)
ap.add_argument("--opt_beginanneal", type=int, default=-1)
ap.add_argument("--opt_fastanneal", action="store_true")
ap.add_argument("--opt_diracq", action="store_true")
ap.add_argument("--opt_sigmoidvar", action="store_true")
ap.add_argument("--opt_pvarbound", type=float, default=0.)
ap.add_argument("--opt_interpretability", action="store_true")
ap.add_argument("--opt_zeroprior", action="store_true")
ap.add_argument("--opt_disentangle", action="store_true")

# Paths
ap.add_argument("--model_path",
                default="/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/checkpoints/ebm.pt")
ap.add_argument("--result_path",
                default="/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/checkpoints/ebm.result")
OPTS.parse(ap)


OPTS.model_path = OPTS.model_path.replace(DATA_ROOT, OPTS.root)
OPTS.result_path = OPTS.result_path.replace(DATA_ROOT, OPTS.root)

if envswitch.who() == "shu":
    OPTS.model_path = os.path.join(DATA_ROOT, os.path.basename(OPTS.model_path))
    OPTS.result_path = os.path.join(DATA_ROOT, os.path.basename(OPTS.result_path))
    OPTS.fixbug1 = True
    OPTS.fixbug2 = True

if envswitch.who() == "jason_prince":
    OPTS.model_path = os.path.join(HOME_DIR, "checkpoints", "ebm", OPTS.dtok, os.path.basename(OPTS.model_path))
    OPTS.result_path = os.path.join(HOME_DIR, "checkpoints", "ebm", OPTS.dtok, os.path.basename(OPTS.result_path))
    os.makedirs(os.path.dirname(OPTS.model_path), exist_ok=True)

# Determine the number of GPUs to use
horovod_installed = importlib.util.find_spec("horovod") is not None
if envswitch.who() != "shu":
    horovod_installed = False
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
            task = Task.init(
                project_name="lanmt2", task_name=OPTS.result_tag, auto_connect_arg_parser=False,
                output_uri=OPTS.root)
            task.connect(ap)
            task.set_random_seed(OPTS.seed)
            task.set_output_model_id(OPTS.model_tag)
            OPTS.trains_task = task
        except:
            pass
        if envswitch.who() != "shu":
            tb_str = "{}_lat{}_noise{}_lr{}".format(OPTS.modeltype, OPTS.latentdim, OPTS.noise, OPTS.ebm_lr)
            if OPTS.train_sgd_steps > 0:
                tb_str += "_imit{}".format(OPTS.train_sgd_steps)
            tb_logdir = os.path.join(HOME_DIR, "tensorboard", "ebm", "{}_cassio".format(OPTS.dtok), tb_str)
            for logdir in [tb_logdir+"_train", tb_logdir+"_dev"]:
                os.makedirs(logdir, exist_ok=True)
        else:
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
if OPTS.fp16:
    print("fp16 option is not ready")
    sys.exit()

# Define dataset
if OPTS.distill:
    tgt_corpus = distilled_tgt_corpus
else:
    tgt_corpus = train_tgt_corpus
n_valid_samples = 5000 if OPTS.finetune else 200
if OPTS.train:
    #if envswitch.who() != "shu":
    #    OPTS.batchtokens = 2048
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
    hidden_size=OPTS.hiddensz,
    embed_size=OPTS.embedsz,
    n_att_heads=OPTS.heads,
    shard_size=OPTS.shard,
    seed=OPTS.seed
)

lanmt_options = basic_options.copy()
lanmt_options.update(dict(
    encoder_layers=5,
    prior_layers=OPTS.priorl,
    q_layers=OPTS.priorl,
    decoder_layers=OPTS.decoderl,
    latent_dim=OPTS.latentdim,
    KL_budget=0. if OPTS.finetune else OPTS.klbudget,
    budget_annealing=OPTS.annealbudget,
    max_train_steps=training_maxsteps,
    fp16=OPTS.fp16
))

nmt = LANMTModel2(**lanmt_options)
if OPTS.scorenet:
    OPTS.shard = 0
    lanmt_model_path = envswitch.load("lanmt_path")
    assert os.path.exists(lanmt_model_path)
    nmt.load(lanmt_model_path)
    if is_root_node():
        print ("Successfully loaded LANMT: {}".format(lanmt_model_path))
    if torch.cuda.is_available():
        nmt.cuda()
    from lib_score_matching6 import LatentScoreNetwork6
    ScoreNet = LatentScoreNetwork6
    # Force to use a specified network
    if OPTS.modelclass == "shunet5":
        from lib_score_matching5_shu import LatentScoreNetwork5
        ScoreNet = LatentScoreNetwork5
    if OPTS.modelclass == "shunet6":
        from lib_score_matching6_shu import LatentScoreNetwork6
        ScoreNet = LatentScoreNetwork6

    nmt = ScoreNet(
        nmt,
        hidden_size=OPTS.hiddensz,
        latent_size=OPTS.latentdim,
        noise=OPTS.noise,
        train_sgd_steps=OPTS.train_sgd_steps,
        train_step_size=OPTS.train_step_size,
        train_delta_steps=OPTS.train_delta_steps,
        modeltype=OPTS.modeltype,
        ebm_useconv=OPTS.ebm_useconv,
        direction_n_layers=OPTS.direction_n_layers,
        magnitude_n_layers=OPTS.magnitude_n_layers,
    )

def postprocess(targets, tgt_vocab):
    target_tokens = targets.cpu().numpy()[0].tolist()
    # target_tokens = [t for t in target_tokens if t > 2]
    target_words = tgt_vocab.decode(target_tokens)
    #target_sent = " ".join(target_words)
    return target_words

def norm2(x1, x2, mask):
    diff = (x1 - x2) ** 2
    diff = diff * mask[:, :, None]
    diff = diff.sum(2).sqrt()
    #return diff.sum(1) / mask.sum(1)
    return torch.masked_select(diff.view(-1), mask.view(-1).byte())

def norm(x1, x2, mask):
    diff = (x1 - x2) ** 2
    diff = diff * mask[:, :, None]
    diff = diff.sum(2).sum(1).sqrt()
    diff = diff / mask.sum(1)
    return diff

# Translate using only one GPU
torch.manual_seed(OPTS.seed)
# Load the autoregressive model for rescoring if neccessary
model_path = OPTS.model_path
model_path = "/scratch/yl1363/lanmt-ebm/checkpoints/ebm/iwslt16_deen/ebm_batchtokens-4092_distill_ebm_lr-0.0003_losstype-original_modeltype-fakegrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-0.8.pt"
#model_path = "/scratch/yl1363/lanmt-ebm/checkpoints/ebm/wmt16_roen/ebm_batchtokens-8192_direction_n_layers-6_distill_dtok-wmt16_roen_fixbug2_modeltype-fakegrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-1.0-bestbest.pt"
if not os.path.exists(model_path):
    print("Cannot find model in {}".format(model_path))
    sys.exit()
nmt.load(model_path)
print ("Successfully loaded EBM: {}".format(model_path))
if torch.cuda.is_available():
    nmt.cuda()
nmt.train(False)
if OPTS.scorenet:
    scorenet = nmt
    OPTS.scorenet = scorenet
    scorenet.train(False)
    nmt = scorenet.nmt()
    nmt.train(False)
src_vocab = Vocab(src_vocab_path)
tgt_vocab = Vocab(tgt_vocab_path)

src_lines = open(test_src_corpus).readlines()
trg_lines = open(test_tgt_corpus).readlines()

sorted_line_ids = np.argsort([len(l.split()) for l in src_lines])
i = 0
n_iter = 8
delta_norms_p = [[] for ii in range(n_iter)]
delta_norms_q = [[] for ii in range(n_iter)]
sgd_norms_p = [[] for ii in range(n_iter)]
sgd_norms_q = [[] for ii in range(n_iter)]

latent_dim = OPTS.latentdim
while i < len(src_lines):
    # Make a batch
    batch_src_lines, batch_trg_lines = [], []
    max_src_len, max_trg_len = 0, 0
    while len(batch_src_lines) * max_src_len < OPTS.Tbatch_size:
        line_id = sorted_line_ids[i]
        src_line, trg_line = src_lines[line_id], trg_lines[line_id]
        src_length, trg_length = len(src_line.split()), len(trg_line.split())
        batch_src_lines.append(src_line); batch_trg_lines.append(trg_line)
        if src_length > max_src_len:
            max_src_len = src_length
        if trg_length > max_trg_len:
            max_trg_len = trg_length
        i += 1
        if i >= len(src_lines):
            break
    x = np.zeros((len(batch_src_lines), max_src_len + 2), dtype="long")
    y = np.zeros((len(batch_trg_lines), max_trg_len + 2), dtype="long")
    for j, src_line in enumerate(batch_src_lines):
        src_tokens = src_vocab.encode("<s> {} </s>".format(src_line.strip()).split())
        trg_tokens = tgt_vocab.encode("<s> {} </s>".format(trg_line.strip()).split())
        x[j, :len(src_tokens)] = src_tokens
        y[j, :len(trg_tokens)] = trg_tokens
    x = torch.tensor(x)
    y = torch.tensor(y)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    x_mask = nmt.to_float(torch.ne(x, 0)).cuda()
    y_mask = nmt.to_float(torch.ne(y, 0)).cuda()
    y_length = y_mask.size(1)
    x_states = nmt.embed_layer(x)
    x_states = nmt.x_encoder(x_states, x_mask)
    with torch.no_grad() if OPTS.modeltype == "fakegrad" else suppress():
        y_states = nmt.embed_layer(y)
        q_states = nmt.q_encoder_xy(y_states, y_mask, x_states, x_mask)
        q_prob = nmt.q_hid2lat(q_states)
        q_mean = q_prob[..., :latent_dim]

        _, p_mean, _, _ = nmt.translate(x, refine_steps=0, y_mask=y_mask)
        z_delta_list = nmt.delta_refine_batch(p_mean, y_mask, x_states, x_mask, n_iter=n_iter)

        z_sgd_list, z_score_list = scorenet.energy_sgd_batch(p_mean, y_mask, x_states, x_mask, n_iter=n_iter, step_size=1.0)

        for idx, z_delta in enumerate(z_delta_list):
            delta_norms_p[idx].append(norm(z_delta, p_mean, y_mask))
            delta_norms_q[idx].append(norm(z_delta, q_mean, y_mask))

        for idx, z_sgd in enumerate(z_sgd_list):
            sgd_norms_p[idx].append(norm(z_sgd, p_mean, y_mask))
            sgd_norms_q[idx].append(norm(z_sgd, q_mean, y_mask))

print ("Norm from p_mean")
for idx, delta_norm in enumerate(delta_norms_p):
    zz = torch.cat(delta_norm, dim=0)
    print ("delta", idx, "{:.3f}".format(zz.mean().item()))
for idx, sgd_norm in enumerate(sgd_norms_p):
    zz = torch.cat(sgd_norm, dim=0)
    print ("score", idx, "{:.3f}".format(zz.mean().item()))

print ("Norm from q_mean")
for idx, delta_norm in enumerate(delta_norms_q):
    zz = torch.cat(delta_norm, dim=0)
    print ("delta", idx, "{:.3f}".format(zz.mean().item()))
for idx, sgd_norm in enumerate(sgd_norms_q):
    zz = torch.cat(sgd_norm, dim=0)
    print ("score", idx, "{:.3f}".format(zz.mean().item()))

import ipdb; ipdb.set_trace()
print (1)
