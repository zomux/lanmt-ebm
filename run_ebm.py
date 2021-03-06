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
from nmtlab.evaluation import MosesBLEUEvaluator, SacreBLEUEvaluator
from collections import defaultdict
import numpy as np
from argparse import ArgumentParser
from contextlib import suppress

from lib_lanmt_model2 import LANMTModel2
from lib_rescoring import load_rescoring_transformer
from lib_envswitch import envswitch
from datasets import get_dataset_paths
from utils import short_tag
from lib_rescoring_fairseq import load_rescoring_transformer

DATA_ROOT = "/misc/vlgscratch4/ChoGroup/jason/corpora/iwslt/iwslt16_ende"
TRAINING_MAX_TOKENS = 60

# Shu paths
envswitch.register("shu", "data_root", "{}/data/wmt14_ende_fair".format(os.getenv("HOME")))
#envswitch.register("jason", "data_root", "/misc/vlgscratch4/ChoGroup/jason/corpora/iwslt/iwslt16_ende")
envswitch.register("jason", "data_root", "/misc/vlgscratch4/ChoGroup/jason/corpora/wmt/wmt14/wmt14_ende_fair")
#envswitch.register("jason_prince", "data_root", "/scratch/yl1363/corpora/iwslt/iwslt16_ende")
#envswitch.register("jason_prince", "data_root", "/scratch/yl1363/corpora/wmt/wmt16/en_ro")
envswitch.register("jason_prince", "data_root", "/scratch/yl1363/corpora/wmt/wmt14/wmt14_ende_fair")

envswitch.register("jason_prince", "home_dir", "/scratch/yl1363/lanmt-ebm")
envswitch.register("jason", "home_dir", "/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm")

DATA_ROOT = envswitch.load("data_root", default=DATA_ROOT)
HOME_DIR = envswitch.load("home_dir", default=DATA_ROOT)
envswitch.register(
    "shu", "lanmt_path",
    os.path.join(DATA_ROOT,
        # "lanmt_anne.ptalbudget_batchtokens-8192_distill_dtok-wmt14_fair_ende_fastanneal_longertrain.pt"
        "lanmt_annealbudget_batchtokens-8192_distill_dtok-wmt14_fair_ende_embedsz-512_fastanneal_heads-8_hiddensz-512_x5longertrain.pt.bak"
     )
)

envswitch.register(
    "shu", "wmt16_roen_lvm",
    "{}/data/jason_checkpoints/wmt16_roen/lanmt_annealbudget_batchtokens-8192_distill_dtok-wmt16_roen_fastanneal_longertrain.pt".format(os.getenv("HOME"))
)

envswitch.register(
    "shu", "wmt16_roen_ebm",
    "{}/data/jason_checkpoints/wmt16_roen/ebm_batchtokens-8192_direction_n_layers-6_distill_dtok-wmt16_roen_fixbug2_modeltype-fakegrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-1.0.pt".format(os.getenv("HOME"))
)

envswitch.register(
    "shu", "iwslt16_deen_lvm",
    "{}/data/jason_checkpoints/iwslt16_deen/lanmt_annealbudget_batchtokens-4092_distill_dtok-iwslt16_deen_tied.pt".format(os.getenv("HOME"))
)

envswitch.register(
    "shu", "iwslt16_deen_ebm",
    "{}/data/jason_checkpoints/iwslt16_deen/ebm_batchtokens-4092_distill_ebm_lr-0.0003_losstype-original_modeltype-fakegrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-0.8.pt".format(os.getenv("HOME"))
)

envswitch.register(
    #"jason", "lanmt_path", "/misc/vlgscratch4/ChoGroup/jason/lanmt/checkpoints/lanmt_annealbudget_batchtokens-4092_distill_dtok-iwslt16_deen_tied.pt"
    #"jason", "lanmt_path", "/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/checkpoints/lvm/iwslt16_deen/lanmt_annealbudget_batchtokens-4092_distill_fastanneal_fixbug2_latentdim-2_lr-0.0003.pt"
    "jason", "lanmt_path", "/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/checkpoints/lvm/wmt14_ende_fair/wmt14_ende_base_v3.pt"
)
envswitch.register(
    #"jason_prince", "lanmt_path", "/scratch/yl1363/lanmt-ebm/checkpoints_lanmt/lanmt_annealbudget_batchtokens-4092_distill_dtok-iwslt16_deen_tied.pt"
    #"jason_prince", "lanmt_path", "/scratch/yl1363/lanmt-ebm/checkpoints/lvm/iwslt16_deen/lanmt_annealbudget_batchtokens-4092_distill_fastanneal_fixbug2_latentdim-2_lr-0.0003.pt"
    #"jason_prince", "lanmt_path", "/scratch/yl1363/lanmt-ebm/checkpoints/lvm/wmt16_roen/lanmt_annealbudget_batchtokens-8192_distill_dtok-wmt16_roen_fastanneal_longertrain.pt"
    "jason_prince", "lanmt_path", "/scratch/yl1363/lanmt-ebm/checkpoints/lvm/wmt14_ende_fair/wmt14_ende_base_v3.pt"
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
ap.add_argument("--fix_layers", action="store_true")
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
ap.add_argument("--opt_x5longertrain", action="store_true")

# Options for LANMT
ap.add_argument("--opt_priorl", type=int, default=6, help="layers for each z encoder")
ap.add_argument("--opt_decoderl", type=int, default=6, help="number of decoder layers")
ap.add_argument("--opt_latentdim", default=8, type=int, help="dimension of latent variables")

# Options for EBM
ap.add_argument("--opt_ebm_lr", default=0.001, type=float)
ap.add_argument("--opt_ebm_useconv", action="store_true")
ap.add_argument("--opt_direction_n_layers", default=4, type=int)
ap.add_argument("--opt_magnitude_n_layers", default=4, type=int)
ap.add_argument("--opt_noise", default=1.0, type=float)
ap.add_argument("--opt_train_sgd_steps", default=0, type=int)
ap.add_argument("--opt_train_step_size", default=0.0, type=float)
ap.add_argument("--opt_train_delta_steps", default=0, type=int)
ap.add_argument("--opt_clipnorm", action="store_true", help="clip the gradient norm")
ap.add_argument("--opt_modeltype", default="whichgrad", type=str)
ap.add_argument("--opt_ebmtype", default="transformer", type=str)
ap.add_argument("--opt_losstype", default="-", type=str)
ap.add_argument("--opt_modelclass", default="", type=str)
ap.add_argument("--opt_fin", default="delta", type=str)
ap.add_argument("--opt_corrupt", action="store_true")
ap.add_argument("--opt_Tsgd_steps", default=1, type=int)
ap.add_argument("--opt_Tstep_size", default=0.8, type=float, help="step size for EBM SGD")
ap.add_argument("--opt_Treport_elbo", action="store_true")
ap.add_argument("--opt_Tline_search", action="store_true")
ap.add_argument("--opt_Tcluster", action="store_true")
ap.add_argument("--opt_decgrad", action="store_true", help="use decoder gradient as target of score matching")
ap.add_argument("--opt_refine_from_mean", action="store_true")
ap.add_argument("--opt_deltasteps", type=int, default=2)

# Decoding options
ap.add_argument("--opt_Twithout_ebm", action="store_true", help="without using EBM")
ap.add_argument("--opt_Tprint_lens", action="store_true", help="print length")
ap.add_argument("--opt_Tsearch_len", default=1, type=int, help="search for multiple length")
ap.add_argument("--opt_Tsearch_lat", default=1, type=int, help="search for multiple length")
ap.add_argument("--opt_Tsearch_stddev", default=1.0, type=float, help="search for multiple length")
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
ap.add_argument("--opt_Tbatch_size", default=8000, type=int, help="batch size for batch translate")
ap.add_argument("--opt_Tpartial_refine", action="store_true")

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
result_dir = os.path.join(DATA_ROOT, "results")
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
OPTS.result_path = "{}/{}.result".format(result_dir, short_tag(OPTS.result_tag))

if envswitch.who() == "shu":
    OPTS.model_path = os.path.join(DATA_ROOT, os.path.basename(OPTS.model_path))
    # OPTS.result_path = os.path.join(DATA_ROOT, os.path.basename(OPTS.result_path))
    OPTS.fixbug1 = True
    if OPTS.dtok != "iwslt16_deen":
        OPTS.fixbug2 = True
else:
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
            tb_logdir = os.path.join(HOME_DIR, "tensorboard", "ebm", OPTS.dtok, OPTS.model_tag)
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
if OPTS.x5longertrain:
    training_maxsteps = int(training_maxsteps * 5)

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
    prior_layers=OPTS.priorl, decoder_layers=OPTS.decoderl,
    q_layers=OPTS.priorl,
    latent_dim=OPTS.latentdim,
    KL_budget=0. if OPTS.finetune else OPTS.klbudget,
    budget_annealing=OPTS.annealbudget,
    max_train_steps=training_maxsteps if OPTS.modeltype != "realgrad" else training_maxsteps * 2,
    fp16=OPTS.fp16
))

nmt = LANMTModel2(**lanmt_options)
if OPTS.scorenet:
    OPTS.shard = 0
    lanmt_model_path = envswitch.load("lanmt_path")
    if OPTS.dtok == "wmt16_roen":
        lanmt_model_path = envswitch.load("wmt16_roen_lvm", default=lanmt_model_path)
    elif OPTS.dtok == "iwslt16_deen":
        lanmt_model_path = envswitch.load("iwslt16_deen_lvm", default=lanmt_model_path)
    print("lanmt_model_path", lanmt_model_path)
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
    if OPTS.modelclass == "denoise6":
        from lib_score_matching6_denoise import LatentScoreNetwork6
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

# Training
if OPTS.train or OPTS.all:
    # Training code
    if OPTS.finetune and not OPTS.scorenet:
        n_valid_per_epoch = 20
        scheduler = SimpleScheduler(max_epoch=1)
    elif OPTS.scorenet:
        n_valid_per_epoch = 10
        #scheduler = SimpleScheduler(max_epoch=5 if envswitch.who() == "shu" else 200)
        training_maxsteps = 60000 if envswitch.who() == "shu" else training_maxsteps
        scheduler = TransformerScheduler(warm_steps=training_warmsteps, max_steps=training_maxsteps)
    else:
        scheduler = TransformerScheduler(warm_steps=training_warmsteps, max_steps=training_maxsteps)
    if OPTS.scorenet and False:
        optimizer = optim.SGD(nmt.parameters(), lr=0.001)
    else:
        optimizer = optim.Adam(nmt.parameters(), lr=OPTS.ebm_lr, betas=(0.9, 0.98), eps=1e-9)
    trainer = MTTrainer(
        nmt, dataset, optimizer,
        scheduler=scheduler, multigpu=gpu_num > 1,
        using_horovod=horovod_installed)
    OPTS.trainer = trainer
    if is_root_node():
        print("TENSORBOARD : ", tb_logdir)
    trainer.configure(
        save_path=OPTS.model_path,
        n_valid_per_epoch=n_valid_per_epoch,
        criteria="loss" if OPTS.scorenet else "loss",
        comp_fn=min if OPTS.scorenet else min,
        tensorboard_logdir=tb_logdir,
        clip_norm=0.1 if OPTS.clipnorm else 0
    )
    # if OPTS.fp16:
    #     from apex import amp
    #     model, optimizer = amp.initialize(nmt, optimizer, opt_level="O3")
    if OPTS.finetune and not OPTS.scorenet:
        pretrain_path = OPTS.model_path.replace("_finetune", "")
        if is_root_node():
            print("loading model:", pretrain_path)
        assert os.path.exists(pretrain_path)
        nmt.load(pretrain_path)
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
    # Load the autoregressive model for rescoring if neccessary
    if OPTS.Tteacher_rescore:
        print("loading teacher fairseq model...")
        if OPTS.dtok == "wmt14_fair_ende":
            fairseq_path = "{}/wmt14_ende_fairseq".format(OPTS.root)
        elif OPTS.dtok == "wmt16_roen":
            fairseq_path = "{}/wmt16_roen_fairseq".format(OPTS.root)
        elif OPTS.dtok == "iwslt16_deen":
            fairseq_path = "{}/iwslt16_deen_fairseq".format(OPTS.root)
        else:
            raise NotImplementedError
        load_rescoring_transformer(src_vocab_path, tgt_vocab_path, fairseq_path)
    # Load models for test >>>
    model_path = OPTS.model_path
    if OPTS.dtok == "wmt16_roen":
        model_path = envswitch.load("wmt16_roen_ebm", default=model_path)
    elif OPTS.dtok == "iwslt16_deen":
        model_path = envswitch.load("iwslt16_deen_ebm", default=model_path)
    if OPTS.modeltype == "realgrad":
        model_path = model_path.replace("fakegrad", "realgrad")
        if OPTS.dtok == "iwslt16_deen":
            model_path = model_path.replace("_train_sgd_steps-1_train_step_size-0.8", "")
    # <<<
    if envswitch.who() != "shu":
        #model_path = "/scratch/yl1363/lanmt-ebm/checkpoints/ebm/iwslt16_deen/ebm_batchtokens-4092_distill_ebm_lr-0.0003_losstype-original_modeltype-fakegrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-0.8.pt"
        #model_path = "/scratch/yl1363/lanmt-ebm/checkpoints/ebm/iwslt16_deen/ebm_batchtokens-4092_distill_ebm_lr-0.0003_losstype-original_modeltype-realgrad_scorenet_train_delta_steps-4.pt"

        #model_path = "/scratch/yl1363/lanmt-ebm/checkpoints/ebm/wmt16_roen/ebm_batchtokens-8192_direction_n_layers-6_distill_dtok-wmt16_roen_fixbug2_modeltype-fakegrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-1.0-bestbest.pt"
        #model_path = "/scratch/yl1363/lanmt-ebm/checkpoints/ebm/wmt16_roen/ebm_batchtokens-8192_direction_n_layers-6_distill_dtok-wmt16_roen_fixbug2_modeltype-realgrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-1.0.pt"

        model_path = "/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/checkpoints/ebm/wmt14_fair_ende/ebm_batchtokens-8192_direction_n_layers-6_distill_dtok-wmt14_fair_ende_ebm_lr-0.0003_embedsz-512_fixbug2_heads-8_hiddensz-512_modeltype-fakegrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-1.0.pt"

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
    result_path = OPTS.result_path
    # Read data
    lines = open(test_src_corpus).readlines()
    ref_lens = [len(l.strip().split()) for l in open(test_tgt_corpus)]
    decode_times = []
    if OPTS.profile:
        lines = lines * 10
    if OPTS.test_fix_length > 0:
        lines = [l for l in lines if len(l.split()) == OPTS.test_fix_length]
        if not lines:
            raise SystemError
        lines = [lines[0]] * 300
    if OPTS.Tprint_lens:
        OPTS.length_collector = []
    trains_stop_stdout_monitor()
    with open(OPTS.result_path, "w") as outf:
        for i, line in enumerate(lines):
            # Make a batch
            tokens = src_vocab.encode("<s> {} </s>".format(line.strip()).split())
            x = torch.tensor([tokens])
            if torch.cuda.is_available():
                x = x.cuda()
            start_time = time.time()

            if OPTS.scorenet:
                with torch.no_grad() if OPTS.modeltype == "fakegrad" else suppress():
                    targets, _, _ = scorenet.translate(
                        x, n_iter=OPTS.Tsgd_steps, step_size=OPTS.Tstep_size, line_search=OPTS.Tline_search, noise=OPTS.Tsearch_stddev)
                    #targets, _, _ = nmt.translate(x, refine_steps=OPTS.Tsgd_steps)
            else:
                targets, _, _ = nmt.translate(x, refine_steps=OPTS.Tsgd_steps)

            target_tokens = targets.cpu().numpy()[0].tolist()
            if targets is None:
                target_tokens = [2, 2, 2]
            # Record decoding time
            end_time = time.time()
            decode_times.append((end_time - start_time) * 1000.)
            # Convert token IDs back to words
            target_tokens = [t for t in target_tokens if t > 2]
            target_words = tgt_vocab.decode(target_tokens)
            target_sent = " ".join(target_words)
            outf.write(target_sent + "\n")
            sys.stdout.write("\rtranslating: {:.1f}%  ".format(float(i) * 100 / len(lines)))
            sys.stdout.flush()
    #import ipdb; ipdb.set_trace()
    sys.stdout.write("\n")
    trains_restore_stdout_monitor()
    print("Average decoding time: {:.0f}ms, std: {:.0f}".format(np.mean(decode_times), np.std(decode_times)))
    if OPTS.Tprint_lens:
        print("Predicted length dumped")
        open("/tmp/length_collector.txt", "w").write(str(OPTS.length_collector))

# Translate multiple sentences in batch
if OPTS.batch_test:
    # Translate using only one GPU
    if not is_root_node():
        sys.exit()
    torch.manual_seed(OPTS.seed)
    if OPTS.Tlatent_search:
        print("--opt_Tlatent_search is not supported in batch test mode right now. Try to implement it.")
    # Load trained model
    model_path = OPTS.model_path
    if envswitch.who() != "shu":
        #model_path = "/scratch/yl1363/lanmt-ebm/checkpoints/ebm/iwslt16_deen/ebm_batchtokens-4092_distill_ebm_lr-0.0003_losstype-original_modeltype-fakegrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-0.8.pt"
        #model_path = "/scratch/yl1363/lanmt-ebm/checkpoints/ebm/iwslt16_deen/ebm_batchtokens-4092_distill_ebm_lr-0.0003_losstype-original_modeltype-realgrad_scorenet_train_delta_steps-4.pt"

        #model_path = "/scratch/yl1363/lanmt-ebm/checkpoints/ebm/wmt16_roen/ebm_batchtokens-8192_direction_n_layers-6_distill_dtok-wmt16_roen_fixbug2_modeltype-fakegrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-1.0-bestbest.pt"
        #model_path = "/scratch/yl1363/lanmt-ebm/checkpoints/ebm/wmt16_roen/ebm_batchtokens-8192_direction_n_layers-6_distill_dtok-wmt16_roen_fixbug2_modeltype-realgrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-1.0.pt"

        #model_path = "/scratch/yl1363/lanmt-ebm/checkpoints/ebm/wmt14_fair_ende/ebm_batchtokens-8192_direction_n_layers-6_distill_dtok-wmt14_fair_ende_ebm_lr-0.0003_embedsz-512_fixbug2_heads-8_hiddensz-512_modeltype-fakegrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-1.0.pt"
        model_path = "/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/checkpoints/ebm/wmt14_fair_ende/ebm_batchtokens-8192_direction_n_layers-6_distill_dtok-wmt14_fair_ende_ebm_lr-0.0003_embedsz-512_fixbug2_heads-8_hiddensz-512_modeltype-fakegrad_scorenet_train_delta_steps-4_train_sgd_steps-1_train_step_size-1.0.pt"
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
    if OPTS.Tteacher_rescore:
        if OPTS.dtok == "wmt14_fair_ende":
            load_rescoring_transformer(
                "{}/wmt14_fair_en.vocab".format(DATA_ROOT),
                "{}/wmt14_fair_de.vocab".format(DATA_ROOT)
            )
        else:
            raise NotImplementedError
    src_vocab = Vocab(src_vocab_path)
    tgt_vocab = Vocab(tgt_vocab_path)
    result_path = OPTS.result_path
    # Read data
    batch_test_size = OPTS.Tbatch_size
    lines = open(test_src_corpus).readlines()
    sorted_line_ids = np.argsort([len(l.split()) for l in lines])
    start_time = time.time()
    output_tokens = []
    elbos, marginals, objs = [], [], []
    i = 0
    while i < len(lines):
        # Make a batch
        batch_lines = []
        max_len = 0
        while len(batch_lines) * max_len < OPTS.Tbatch_size:
            line_id = sorted_line_ids[i]
            line = lines[line_id]
            length = len(line.split())
            batch_lines.append(line)
            if length > max_len:
                max_len = length
            i += 1
            if i >= len(lines):
                break
        x = np.zeros((len(batch_lines), max_len + 2), dtype="long")
        for j, line in enumerate(batch_lines):
            tokens = src_vocab.encode("<s> {} </s>".format(line.strip()).split())
            x[j, :len(tokens)] = tokens
        x = torch.tensor(x)
        if torch.cuda.is_available():
            x = x.cuda()

        if OPTS.scorenet:
            with torch.no_grad() if OPTS.modeltype == "fakegrad" else suppress():
                targets, z, _, obj = scorenet.translate(
                    x, n_iter=OPTS.Tsgd_steps, step_size=OPTS.Tstep_size, line_search=OPTS.Tline_search, report=True)
                #targets, z, _, obj = nmt.translate(x, refine_steps=OPTS.Tsgd_steps, report=True)
        else:
            targets, _ = nmt.translate(x, refine_steps=OPTS.Tsgd_steps)
        if envswitch.who() != "shu" and OPTS.Treport_elbo:
            with torch.no_grad():
                elbo, marginal = nmt.compute_elbo_and_marginal(x, targets, n_samples=20)
                elbos.extend( elbo.cpu().numpy().tolist() )
                marginals.extend( marginal.cpu().numpy().tolist() )
                objs.extend( obj.cpu().numpy().tolist() )
        target_tokens = targets.cpu().numpy().tolist()
        output_tokens.extend(target_tokens)
        sys.stdout.write("\rtranslating: {:.1f}%  ".format(float(i) * 100 / len(lines)))
        sys.stdout.flush()
    if envswitch.who() != "shu" and OPTS.Treport_elbo:
        print()
        print("ELBO     =", np.mean(elbos))
        print("Marginal =", np.mean(marginals))
        print("Obj      =", np.mean(objs))
        #import ipdb; ipdb.set_trace()

    #subword_idx = []
    with open(OPTS.result_path, "w") as outf:
        # Record decoding time
        end_time = time.time()
        decode_time = (end_time - start_time)
        # Convert token IDs back to words
        id_token_pairs = list(zip(sorted_line_ids, output_tokens))
        id_token_pairs.sort()
        for _, target_tokens in id_token_pairs:
            target_tokens = [t for t in target_tokens if t > 2]
            target_words = tgt_vocab.decode(target_tokens)
            #subword_idx.append(target_tokens)
            #target_sent = " ".join(target_words)
            target_sent = " ".join(target_words).replace("\n", "").replace("\r", "")
            outf.write(target_sent + "\n")
    sys.stdout.write("\n")
    #import pickle as pkl
    #pkl.dump(subword_idx,
    #         open(os.path.join(HOME_DIR, "analysis", "wmt14_ende_results", "delta_{}.idx".format(OPTS.Tsgd_steps)), "wb"))
    print("Batch decoding time: {:.2f}s".format(decode_time))

# Evaluation of translaton quality
if OPTS.evaluate or OPTS.all:
    # Post-processing
    if is_root_node():
        hyp_path = "/tmp/{}.txt".format(OPTS.model_tag)
        if envswitch.who() != "shu":
            hyp_path = os.path.join(HOME_DIR, "hyp", "{}_{}_{}_{}_{}.txt".format(
                OPTS.Tsgd_steps, OPTS.Tstep_size, OPTS.Tsearch_len, OPTS.Tsearch_lat, OPTS.Tsearch_stddev))
        result_path = OPTS.result_path
        with open(hyp_path, "w") as outf:
            for line in open(result_path):
                # Remove duplicated tokens
                tokens = line.strip().split()
                new_tokens = []
                for tok in tokens:
                    if len(new_tokens) > 0 and tok != new_tokens[-1]:
                        new_tokens.append(tok)
                    elif len(new_tokens) == 0:
                        new_tokens.append(tok)
                new_line = " ".join(new_tokens) + "\n"
                line = new_line
                # Remove sub-word indicator in sentencepiece and BPE
                line = line.replace("@@ ", "")
                if "▁" in line:
                    line = line.strip()
                    line = "".join(line.split())
                    line = line.replace("▁", " ").strip() + "\n"
                outf.write(line)
        # Get BLEU score
        if "iwslt" in OPTS.dtok:
            evaluator = SacreBLEUEvaluator(ref_path=ref_path, tokenizer="intl", lowercase=True)
        elif "wmt" in OPTS.dtok:
            if envswitch.who() == "shu":
                script = "{}/scripts/detokenize.perl".format(os.path.dirname(__file__))
            else:
                script = os.path.join(envswitch.load("home_dir"), "scripts", "detokenize.perl")
            os.system("perl {} < {} > {}.detok".format(script, hyp_path, hyp_path))
            hyp_path = hyp_path + ".detok"
            evaluator = SacreBLEUEvaluator(ref_path=ref_path, tokenizer="intl", lowercase=True)
        else:
            evaluator = MosesBLEUEvaluator(ref_path=ref_path)
        bleu = evaluator.evaluate(hyp_path)
        print("BLEU =", bleu)
        if envswitch.who() == "shu":
            from tensorboardX import SummaryWriter
            tb = SummaryWriter(log_dir=tb_logdir, comment="nmtlab")
            tb.add_scalar("BLEU", bleu)
        if envswitch.who() != "shu" and False:
            bleu_file_path = os.path.join(HOME_DIR, "bleu_file_{}".format(OPTS.modeltype))
            bleu_file = open(bleu_file_path, "a")
            bleu_file.write(
                "{},{},{:.4f}\r\n".format(
                    OPTS.Tsgd_steps, OPTS.Tstep_size, bleu))
            bleu_file.close()
