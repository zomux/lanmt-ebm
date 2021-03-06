#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os, sys
from torch import optim
import importlib
import time
import numpy as np

import torch
from nmtlab import MTTrainer, MTDataset
from nmtlab.utils import OPTS
from nmtlab.models import Transformer
from nmtlab.schedulers import TransformerScheduler
from nmtlab.utils import is_root_node, Vocab
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
# if OPTS.test or OPTS.all:
#     import torch
#     from nmtlab.decoding import BeamTranslator
#     translator = BeamTranslator(nmt, dataset.src_vocab(), dataset.tgt_vocab(), beam_size=OPTS.Tbeam)
#     if OPTS.chkavg:
#         chk_count = 0
#         state_dict = None
#         for i in range(1000):
#             path = OPTS.model_path + ".chk{}".format(i)
#             if os.path.exists(path):
#                 chkpoint = torch.load(path)["model_state"]
#                 if state_dict is None:
#                     state_dict = chkpoint
#                 else:
#                     for key, val in chkpoint.items():
#                         state_dict[key] += val
#                 chk_count += 1
#         for key in state_dict.keys():
#             state_dict[key] /= float(chk_count)
#         if is_root_node():
#             print("Averaged {} checkpoints".format(chk_count))
#         translator.model.load_state_dict(state_dict)
#     else:
#         assert os.path.exists(OPTS.model_path)
#         translator.load(OPTS.model_path)
#     translator.batch_translate(test_src_corpus, OPTS.result_path)
#     if is_root_node():
#         print("[Translation Result]")
#         print(OPTS.result_path)

if OPTS.test or OPTS.all:
    # Translate using only one GPU
    if not is_root_node():
        sys.exit()
    torch.manual_seed(OPTS.seed)
    # Load the autoregressive model for rescoring if neccessary
    if OPTS.Tteacher_rescore:
        assert os.path.exists(pretrained_autoregressive_path)
        load_rescoring_transformer(basic_options, pretrained_autoregressive_path)
    model_path = OPTS.model_path
    if not os.path.exists(model_path):
        print("Cannot find model in {}".format(model_path))
        sys.exit()
    # model_path = "{}/basemodel_wmt14_ende_x5longertrain_v2.pt.bak".format(DATA_ROOT)
    nmt.load(model_path)
    if torch.cuda.is_available():
        nmt.cuda()
    nmt.train(False)
    from nmtlab.decoding import BeamTranslator
    translator = BeamTranslator(nmt, dataset.src_vocab(), dataset.tgt_vocab(), beam_size=OPTS.Tbeam)
    src_vocab = Vocab(src_vocab_path)
    tgt_vocab = Vocab(tgt_vocab_path)
    result_path = OPTS.result_path
    # Read data
    lines = open(test_src_corpus).readlines()
    latent_candidate_num = OPTS.Tcandidate_num if OPTS.Tlatent_search else None
    decode_times = []
    if OPTS.profile:
        lines = lines * 10
    # lines = lines[:100]
    # trains_stop_stdout_monitor()
    with open(OPTS.result_path, "w") as outf:
        for i, line in enumerate(lines):
            # Make a batch
            tokens = src_vocab.encode("<s> {} </s>".format(line.strip()).split())
            x = torch.tensor([tokens])
            if torch.cuda.is_available():
                x = x.cuda()
            start_time = time.time()
            # with torch.no_grad() if not OPTS.scorenet else nullcontext():
                # Predict latent and target words from prior
            target_sent, _ = translator.translate(line)
            if target_sent is None:
                target_sent = ""
            # target_words = targets[0].cpu()[0].numpy().tolist()
            # Record decoding time
            end_time = time.time()
            decode_times.append((end_time - start_time) * 1000.)
            # Convert token IDs back to words
            # target_sent = " ".join(target_words)
            outf.write(target_sent + "\n")
            sys.stdout.write("\rtranslating: {:.1f}%  ".format(float(i) * 100 / len(lines)))
            sys.stdout.flush()
    sys.stdout.write("\n")
    # trains_restore_stdout_monitor()
    print("Average decoding time: {:.0f}ms, std: {:.0f}".format(np.mean(decode_times), np.std(decode_times)))


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
        if "wmt" in OPTS.dtok:
            script = "{}/scripts/detokenize.perl".format(os.path.dirname(__file__))
            os.system("perl {} < {} > {}.detok".format(script, hyp_path, hyp_path))
            hyp_path = hyp_path + ".detok"
            from nmtlab.evaluation import SacreBLEUEvaluator
            evaluator = SacreBLEUEvaluator(ref_path=ref_path, tokenizer="intl", lowercase=True)
        else:
            evaluator = MosesBLEUEvaluator(ref_path=ref_path)
        bleu = evaluator.evaluate(hyp_path)
        print("BLEU =", bleu)


