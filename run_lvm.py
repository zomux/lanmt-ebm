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

TRAINING_MAX_TOKENS = 60

# Shu paths
envswitch.register("shu", "data_root", "{}/data/wmt14_ende_fair".format(os.getenv("HOME")))
envswitch.register("jason_prince", "data_root", "/scratch/yl1363/corpora/iwslt/iwslt16_ende")
envswitch.register("jason", "data_root", "/misc/vlgscratch4/ChoGroup/jason/corpora/iwslt/iwslt16_ende")
envswitch.register("jason_prince", "home_dir", "/scratch/yl1363/lanmt-ebm")
envswitch.register("jason", "home_dir", "/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm")

DATA_ROOT = envswitch.load("data_root")
HOME_DIR = envswitch.load("home_dir")

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
ap.add_argument("--opt_lr", default=0.001, type=float)
ap.add_argument("--opt_clipnorm", action="store_true", help="clip the gradient norm")

# Commmon option for both autoregressive and non-autoregressive models
ap.add_argument("--opt_batchtokens", type=int, default=4096)
ap.add_argument("--opt_hiddensz", type=int, default=256)
ap.add_argument("--opt_embedsz", type=int, default=256)
ap.add_argument("--opt_heads", type=int, default=4)
ap.add_argument("--opt_shard", type=int, default=32)
ap.add_argument("--opt_longertrain", action="store_true")
ap.add_argument("--opt_x3longertrain", action="store_true")

# Options for LANMT
ap.add_argument("--opt_priorl", type=int, default=6, help="layers for each z encoder")
ap.add_argument("--opt_decoderl", type=int, default=6, help="number of decoder layers")
ap.add_argument("--opt_latentdim", default=8, type=int, help="dimension of latent variables")

# Decoding options
ap.add_argument("--opt_distill", action="store_true", help="train with knowledge distillation")
ap.add_argument("--opt_annealbudget", action="store_true", help="switch of annealing KL budget")
ap.add_argument("--opt_fixbug1", action="store_true", help="fix bug in length converter")
ap.add_argument("--opt_fixbug2", action="store_true", help="fix bug in transformer decoder")
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
                default="/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/checkpoints_lvm/lanmt.pt")
ap.add_argument("--result_path",
                default="/misc/vlgscratch4/ChoGroup/jason/lanmt-ebm/checkpoints_lvm/lanmt.result")
OPTS.parse(ap)


OPTS.model_path = OPTS.model_path.replace(DATA_ROOT, OPTS.root)
OPTS.result_path = OPTS.result_path.replace(DATA_ROOT, OPTS.root)

if envswitch.who() == "shu":
    OPTS.model_path = os.path.join(DATA_ROOT, os.path.basename(OPTS.model_path))
    OPTS.result_path = os.path.join(DATA_ROOT, os.path.basename(OPTS.result_path))
    OPTS.fixbug1 = True
    OPTS.fixbug2 = True
else:
    OPTS.model_path = os.path.join(HOME_DIR, "checkpoints", "lvm", OPTS.dtok, os.path.basename(OPTS.model_path))
    OPTS.result_path = os.path.join(HOME_DIR, "checkpoints", "lvm", OPTS.dtok, os.path.basename(OPTS.result_path))
    os.makedirs(os.path.dirname(OPTS.model_path), exist_ok=True)
    #OPTS.fixbug2 = True

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
            tb_logdir = os.path.join(HOME_DIR, "tensorboard", "lvm", OPTS.dtok, OPTS.model_tag)
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
    prior_layers=OPTS.priorl, decoder_layers=OPTS.decoderl,
    latent_dim=OPTS.latentdim,
    KL_budget=0. if OPTS.finetune else OPTS.klbudget,
    budget_annealing=OPTS.annealbudget,
    max_train_steps=200000 if (OPTS.dtok == "iwslt16_deen" and envswitch.who() != "shu") else training_maxsteps,
    fp16=OPTS.fp16
))

nmt = LANMTModel2(**lanmt_options)

# Training
if OPTS.train or OPTS.all:
    # Training code
    if OPTS.finetune:
        n_valid_per_epoch = 20
        scheduler = SimpleScheduler(max_epoch=1)
    else:
        scheduler = TransformerScheduler(warm_steps=training_warmsteps, max_steps=training_maxsteps)
    optimizer = optim.Adam(nmt.parameters(), lr=OPTS.lr, betas=(0.9, 0.98), eps=1e-9)
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
        criteria="neg_elbo",
        comp_fn=min,
        tensorboard_logdir=tb_logdir,
        clip_norm=0.1 if OPTS.clipnorm else 0
    )
    # if OPTS.fp16:
    #     from apex import amp
    #     model, optimizer = amp.initialize(nmt, optimizer, opt_level="O3")
    if OPTS.finetune:
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
        assert os.path.exists(pretrained_autoregressive_path)
        load_rescoring_transformer(basic_options, pretrained_autoregressive_path)
    # Load trained model
    if OPTS.use_pretrain:
        if OPTS.dtok not in PRETRAINED_MODEL_MAP:
            print("The model for {} doesn't exist".format(OPTS.dtok))
        model_path = PRETRAINED_MODEL_MAP[OPTS.dtok]
        print("loading pretrained model in {}".format(model_path))
        OPTS.result_path = OPTS.result_path.replace("lanmt_", "lanmt_pretrain_")
    else:
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
    result_path = OPTS.result_path
    # Read data
    lines = open(test_src_corpus).readlines()
    latent_candidate_num = OPTS.Tcandidate_num if OPTS.Tlatent_search else None
    decode_times = []
    with open(OPTS.result_path, "w") as outf:
        for i, line in enumerate(lines):
            # Make a batch
            tokens = src_vocab.encode("<s> {} </s>".format(line.strip()).split())
            x = torch.tensor([tokens])
            if torch.cuda.is_available():
                x = x.cuda()
            start_time = time.time()
            with torch.no_grad():
                # Predict latent and target words from prior
                targets, _, prior_states = nmt.translate(x)
                target_tokens = targets.cpu().numpy()[0].tolist()
                # Interative inference
                for infer_step in range(OPTS.Trefine_steps):
                    # Sample latent from Q and draw a new target prediction
                    prev_target = tuple(target_tokens)
                    new_latent, _ = nmt.compute_Q(x, targets)
                    targets, _, _ = nmt.translate(x, latent=new_latent, prior_states=prior_states,
                                                  refine_step=infer_step + 1)
                    target_tokens = targets[0].cpu().numpy().tolist()
                    # Early stopping
                    if tuple(target_tokens) == tuple(prev_target):
                        break
            if targets is None:
                target_tokens = [2, 2, 2]
            elif OPTS.Tteacher_rescore:
                scores = OPTS.teacher(x, targets)
                target_tokens = targets[scores.argmax()]
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
    sys.stdout.write("\n")
    print("Average decoding time: {:.0f}ms, std: {:.0f}".format(np.mean(decode_times), np.std(decode_times)))

# Translate multiple sentences in batch
if OPTS.batch_test:
    # Translate using only one GPU
    if not is_root_node():
        sys.exit()
    torch.manual_seed(OPTS.seed)
    if OPTS.Tlatent_search:
        print("--opt_Tlatent_search is not supported in batch test mode right now. Try to implement it.")
    # Load trained model
    if OPTS.use_pretrain:
        if OPTS.dtok not in PRETRAINED_MODEL_MAP:
            print("The model for {} doesn't exist".format(OPTS.dtok))
        model_path = PRETRAINED_MODEL_MAP[OPTS.dtok]
        print("loading pretrained model in {}".format(model_path))
        OPTS.result_path = OPTS.result_path.replace("lanmt_", "lanmt_pretrain_")
    else:
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
    result_path = OPTS.result_path
    # Read data
    batch_test_size = OPTS.Tbatch_size
    lines = open(test_src_corpus).readlines()
    sorted_line_ids = np.argsort([len(l.split()) for l in lines])
    start_time = time.time()
    output_tokens = []
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
        with torch.no_grad():
            # Predict latent and target words from prior
            targets, _, _ = nmt.translate(x, refine_steps=OPTS.Trefine_steps)
        target_tokens = targets.cpu().numpy().tolist()
        output_tokens.extend(target_tokens)
        sys.stdout.write("\rtranslating: {:.1f}%  ".format(float(i) * 100 / len(lines)))
        sys.stdout.flush()

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
            target_sent = " ".join(target_words)
            outf.write(target_sent + "\n")
    sys.stdout.write("\n")
    print("Batch decoding time: {:.2f}s".format(decode_time))

# Evaluation of translaton quality
if OPTS.evaluate or OPTS.all:
    # Post-processing
    if is_root_node():
        hyp_path = "/tmp/namt_hyp.txt"
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
           script = "{}/scripts/detokenize.perl".format(os.path.dirname(__file__))
           os.system("perl {} < {} > {}.detok".format(script, hyp_path, hyp_path))
           hyp_path = hyp_path + ".detok"
           evaluator = SacreBLEUEvaluator(ref_path=ref_path, tokenizer="intl", lowercase=True)
        else:
            evaluator = MosesBLEUEvaluator(ref_path=ref_path)
        bleu = evaluator.evaluate(hyp_path)
        print("BLEU =", bleu)

