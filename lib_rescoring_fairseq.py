#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from nmtlab.models import Transformer
from nmtlab.utils import MapDict, OPTS
from fairseq.models.transformer import TransformerModel
from fairseq.models.fairseq_encoder import EncoderOut
from nmtlab.utils import Vocab

from collections import defaultdict

class FairseqReranker(object):

    def __init__(self, src_vocab_path, tgt_vocab_path,
                 fairseq_path="/home/acb11204eq/data/wmt14_ende_fair/wmt14_ende_fairseq"):
        self.src_vmap = self.build_vocab_map(src_vocab_path, "{}/dict.src.txt".format(fairseq_path))
        self.tgt_vmap = self.build_vocab_map(tgt_vocab_path, "{}/dict.tgt.txt".format(fairseq_path))
        model = TransformerModel.from_pretrained(
            fairseq_path,
            checkpoint_file="{}/checkpoint.pt".format(fairseq_path),
            data_name_or_path=fairseq_path)
        # model.translate("Yesterday , Gut@@ acht &apos;s Mayor gave a clear answer to this question .")
        if torch.cuda.is_available():
            model.cuda()
        self.transformer = model._modules["models"][0]
        self.transformer.train(False)


    def build_vocab_map(self, vocab_path, fairseq_vocab_path):
        hmap = defaultdict(lambda: 3)
        lines = open(fairseq_vocab_path).read().strip().split("\n")
        fairseq_map = defaultdict(lambda: 3)
        for i, w in enumerate(lines):
            w = w.split()[0]
            fairseq_map[w] = i + 4
        lines = open(vocab_path).read().strip().split("\n")
        for i, w in enumerate(lines):
            hmap[i] = fairseq_map[w]
        hmap[0] = 1
        hmap[1] = 2
        hmap[2] = 2
        hmap[3] = 3
        return hmap

    def score(self, src_tokens, tgt_tokens):
         src_tokens = src_tokens[:, 1:]
         assert src_tokens.shape[0] == 1
         unique_tgt_tokens = tgt_tokens.unique(dim=0)
         x = src_tokens[0].cpu().numpy()
         for i in range(x.shape[0]):
             x[i] = self.src_vmap[x[i]]
         y = unique_tgt_tokens.cpu().numpy()
         for r in range(y.shape[0]):
             pad = False
             for c in range(y.shape[1]):
                 if pad:
                    y[r][c] = 1
                 else:
                    if y[r][c] == 2:
                         pad = True
                    y[r][c] = self.tgt_vmap[y[r][c]]
         B = unique_tgt_tokens.shape[0]
         with torch.no_grad():
             x_tensor = torch.tensor(x)[None, :].cuda()
             y_tensor = torch.tensor(y).cuda()
             x_lens = torch.tensor([x_tensor.shape[1]]).cuda()
             y_lens = torch.ne(y_tensor, 1).sum(1) - 1
             # Transformer forward >>>
             encoder_out = self.transformer.encoder(
                 x_tensor,
                 src_lengths=x_lens,
                 return_all_hiddens=False
             )
             encoder_out = EncoderOut(
                 encoder_out.encoder_out.repeat(1, B, 1),
                 encoder_out.encoder_padding_mask.repeat(B, 1),
                 encoder_out.encoder_embedding.repeat(B, 1, 1), None, None, None)
             decoder_out = self.transformer.decoder(
                 y_tensor[:, :-1],
                 encoder_out=encoder_out,
                 src_lengths=x_lens.repeat(B),
                 return_all_hiddens=False,
             )
             logits = decoder_out[0]
             # <<<
             logp = torch.log_softmax(logits, 2)
             _, L, V = logp.shape
             token_logp = logp.view(B*L, V)[torch.arange(B*L), y_tensor[:, 1:].flatten()].view(B, L)
             y_mask = torch.arange(L).unsqueeze(0).repeat(B, 1).cuda() < y_lens[:, None]
             scores = (token_logp * y_mask).sum(1) / y_mask.sum(1)
         return unique_tgt_tokens, scores



def load_rescoring_transformer(src_vocab_path, tgt_vocab_path, fairseq_path):
    OPTS.teacher = FairseqReranker(src_vocab_path, tgt_vocab_path, fairseq_path)
    return OPTS.teacher

if __name__ == '__main__':
    root = "/home/acb11204eq/data/wmt14_ende_fair"
    load_rescoring_transformer(
        "{}/wmt14_fair_en.vocab".format(root),
        "{}/wmt14_fair_de.vocab".format(root)
    )
    x = torch.tensor([[1, 18, 19, 20, 2]]).cuda()
    y = torch.tensor([[1, 19, 50, 5, 20, 2], [1, 12, 6, 20, 2, 0]]).cuda()
    scores = OPTS.teacher.score(x, y)
    print(scores)