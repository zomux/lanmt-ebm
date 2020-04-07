#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This model unifies the training of decoder, latent encoder, latent predictor
"""

from __future__ import division
from __future__ import print_function

import os, sys
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

from lib_horovod import initialize_horovod
from lib_trains import initialize_trains

from lib_independent_ebm import IndependentEnergyMT
from lib_lanmt_model import LANMTModel
from datasets import get_dataset_paths

DATA_ROOT = "./mydata"
PRETRAINED_MODEL_MAP = {
    "wmt14_ende": "{}/shu_trained_wmt14_ende.pt".format(DATA_ROOT),
    "aspec_jaen": "{}/shu_trained_aspec_jaen.pt".format(DATA_ROOT),
}
TRAINING_MAX_TOKENS = 60

ap = ArgumentParser()
ap.add_argument("--root", type=str, default=DATA_ROOT)
ap.add_argument("--all", action="store_true")
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

# Options for LM
ap.add_argument("--opt_corruption", type=str, default="target")
ap.add_argument("--opt_corrupt", type=float, default=0.2)
ap.add_argument("--opt_losstype", type=str, default="single")
ap.add_argument("--opt_modeltype", type=str, default="fakegrad")
ap.add_argument("--opt_enctype", type=str, default="conv")
ap.add_argument("--opt_dectype", type=str, default="conv")
ap.add_argument("--opt_ebmtype", type=str, default="conv")
ap.add_argument("--opt_nrefine", type=int, default=1)
ap.add_argument("--opt_epochs", type=int, default=20)

ap.add_argument("--opt_Tbaseline", action="store_true")

# Paths
ap.add_argument("--model_path",
                default="{}/indp_mt.pt".format(DATA_ROOT))
ap.add_argument("--result_path",
                default="{}/indp_mt.result".format(DATA_ROOT))
OPTS.parse(ap)

OPTS.fixbug1 = True
OPTS.fixbug2 = True
OPTS.model_path = OPTS.model_path.replace(DATA_ROOT, OPTS.root)
OPTS.result_path = OPTS.result_path.replace(DATA_ROOT, OPTS.root)

# Determine the number of GPUs to use
gpu_index, gpu_num = initialize_horovod()

# Trains Logging
tb_logdir = initialize_trains(ap, "IND_EBM_MT", OPTS.result_tag)

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


if OPTS.train or OPTS.all:
    dataset = MTDataset(
        src_corpus=train_src_corpus, tgt_corpus=tgt_corpus,
        src_vocab=src_vocab_path, tgt_vocab=tgt_vocab_path,
        batch_size=OPTS.batchtokens * gpu_num, batch_type="token",
        truncate=truncate_datapoints, max_length=TRAINING_MAX_TOKENS,
        n_valid_samples=500)
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

nmt = IndependentEnergyMT(latent_size=OPTS.latentdim)


# Training
if OPTS.train or OPTS.all:
    # Training code
    scheduler = SimpleScheduler(max_epoch=OPTS.epochs)
    # scheduler = TransformerScheduler(warm_steps=training_warmsteps, max_steps=training_maxsteps)
    lr = 0.0001 * gpu_num / 8
    optimizer = optim.Adam(nmt.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-4)

    trainer = MTTrainer(
        nmt, dataset, optimizer,
        scheduler=scheduler, multigpu=gpu_num > 1,
        using_horovod=gpu_num > 1)
    OPTS.trainer = trainer
    trainer.configure(
        save_path=OPTS.model_path,
        n_valid_per_epoch=n_valid_per_epoch,
        criteria="loss",
        tensorboard_logdir=tb_logdir,
        save_optim_state=False
        # clip_norm=0.1 if OPTS.scorenet else 0
    )
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

    # Load LANMT
    src_vocab = Vocab(src_vocab_path)
    tgt_vocab = Vocab(tgt_vocab_path)
    lanmt_options = dict(
        src_vocab_size=src_vocab.size(),
        tgt_vocab_size=tgt_vocab.size(),
        hidden_size=512, embed_size=512,
        n_att_heads=8,
        prior_layers=6, decoder_layers=6, latent_dim=8
    )
    OPTS.zeroprior = True
    lanmt = LANMTModel(**lanmt_options)
    lanmt.load(os.path.join(OPTS.root,
                            "lanmt_annealbudget_beginanneal-20000_distill_dtok-wmt14_fair_ende_fastanneal_finetune_fixbug1_fixbug2_klbudget-10.0_x3longertrain_zeroprior.pt"))
    lanmt.train(False)
    if torch.cuda.is_available():
        lanmt.cuda()

    # Testing
    lines = open(test_src_corpus).readlines()
    trains_stop_stdout_monitor()
    with open(OPTS.result_path, "w") as outf:
        for i, line in enumerate(lines):
            # Make a batch
            tokens = src_vocab.encode("<s> {} </s>".format(line.strip()).split())
            x = torch.tensor([tokens])
            if torch.cuda.is_available():
                x = x.cuda()
            mask = torch.ne(x, 0).float()
            # Compute base prediction with LANMT
            with torch.no_grad():
                prior_states = lanmt.prior_encoder(x, mask)
                z = torch.zeros((1, x.shape[1], 8), requires_grad=True)
                if torch.cuda.is_available():
                    z = z.cuda()
                latent = lanmt.latent2vector_nn(z)
                targets, _, _ = lanmt.translate(x)
                target_tokens = targets.cpu().numpy()[0].tolist()
            # EBM refinement
            if not OPTS.Tbaseline:
                target_mask = torch.ne(targets, 0).float()
                logits = nmt.compute_logits(x, mask, targets, target_mask)
                targets = logits.argmax(2)
                target_tokens = targets.cpu().numpy()[0].tolist()
            # Convert token IDs back to words
            target_tokens = [t for t in target_tokens if t > 2]
            target_words = tgt_vocab.decode(target_tokens)
            target_sent = " ".join(target_words)
            outf.write(target_sent + "\n")
            sys.stdout.write("\rtranslating: {:.1f}%  ".format(float(i) * 100 / len(lines)))
            sys.stdout.flush()
    sys.stdout.write("\n")
    trains_restore_stdout_monitor()

if OPTS.evaluate or OPTS.all:
    from nmtlab.evaluation.sacre_bleu import SacreBLEUEvaluator
    from tensorboardX import SummaryWriter
    tb = SummaryWriter(log_dir=tb_logdir, comment="nmtlab")
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
        if "wmt" in OPTS.dtok:
            script = "{}/scripts/detokenize.perl".format(os.path.dirname(__file__))
            os.system("perl {} < {} > {}.detok".format(script, hyp_path, hyp_path))
            hyp_path = hyp_path + ".detok"
            evaluator = SacreBLEUEvaluator(ref_path=ref_path, tokenizer="intl", lowercase=True)
        else:
            evaluator = MosesBLEUEvaluator(ref_path=ref_path)
        bleu = evaluator.evaluate(hyp_path)
        print("BLEU =", bleu)
        tb.add_scalar("BLEU", bleu)


