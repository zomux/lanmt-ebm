#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Transformer VAE with a Gaussian prior/approximate posterior and a non-autoregressive decoder.

A few changes to LANMTModel:

- Sharing source embeddings, target embeddings and target linear layer (which produces the logits)
- Length prediction is done once, from the source sentence hidden states, and fixed throughout refinement.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nmtlab.models.transformer import Transformer
from nmtlab.modules.transformer_modules import TransformerEmbedding
from nmtlab.modules.transformer_modules import PositionalEmbedding
from nmtlab.modules.transformer_modules import LabelSmoothingKLDivLoss
from nmtlab.utils import OPTS
from nmtlab.utils import TensorMap
from nmtlab.utils import smoothed_bleu

from lib_lanmt_modules import TransformerEncoder
from lib_lanmt_modules import TransformerCrossEncoder

from lib_envswitch import envswitch


class LANMTModel2(Transformer):

    def __init__(self,
                 encoder_layers=5, 
                 prior_layers=3, decoder_layers=3,
                 q_layers=6,
                 latent_dim=8,
                 KL_budget=1., KL_weight=1.,
                 budget_annealing=True,
                 max_train_steps=100000,
                 tied=False,
                 **kwargs):
        """Create Latent-variable non-autoregressive NMT model.

        Args:
            prior_layers - number of layers in prior p(z|x)
            decoder_layers - number of layers in decoder p(y|z)
            q_layers - number of layers in approximator q(z|x,y)
            latent_dim - dimension of latent variables
            KL_budget - budget of KL divergence
            KL_weight - weight of the KL term,
            budget_annealing - whether anneal the KL budget
            max_train_steps - max training iterations
        """
        self.encoder_layers = encoder_layers
        self.prior_layers = prior_layers
        self.decoder_layers = decoder_layers
        self.q_layers = q_layers
        self.latent_dim = latent_dim
        self.KL_budget = KL_budget
        self.KL_weight = KL_weight
        self.budget_annealing = budget_annealing
        self.max_train_steps = max_train_steps
        self.tied = tied
        if OPTS.finetune:
            self.training_criteria = "BLEU"
        else:
            self.training_criteria = "loss"
        super(LANMTModel2, self).__init__(**kwargs)
        assert self._src_vocab_size == self._tgt_vocab_size

    def prepare(self):
        """Define the modules
        """
        # Embedding layers
        self.embed_layer = TransformerEmbedding(self._tgt_vocab_size, self.embed_size)
        embed_layer = self.embed_layer
        self.pos_embed_layer = PositionalEmbedding(self.hidden_size)
        self.x_encoder = TransformerEncoder(
          None, self.hidden_size, 5 if OPTS.fix_layers else self.encoder_layers)

        # Prior p(z|x)
        self.prior_encoder = TransformerCrossEncoder(
          None, self.hidden_size, 3 if OPTS.fix_layers else self.prior_layers)
        self.p_hid2lat = nn.Linear(self.hidden_size, self.latent_dim * 2)

        # Approximate Posterior q(z|y,x)
        self.q_encoder_xy = TransformerCrossEncoder(
          None, self.hidden_size, 3 if OPTS.fix_layers else self.q_layers)
        self.q_hid2lat = nn.Linear(self.hidden_size, self.latent_dim * 2)

        # Decoder p(y|x,z)
        self.lat2hid = nn.Linear(self.latent_dim, self.hidden_size)
        self.decoder = TransformerCrossEncoder(
          None, self.hidden_size, 3 if OPTS.fix_layers else self.decoder_layers, skip_connect=True)

        # Length prediction
        #self.length_predictor = nn.Linear(self.hidden_size, 100)
        self.length_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 100)
        )

        # Word probability estimator
        self.final_bias = nn.Parameter(torch.randn(self._tgt_vocab_size))
        self.final_bias.requires_grad = True
        final_bias = self.final_bias
        class FinalLinear(object):
          def __call__(self, x):
            return x @ torch.transpose(embed_layer.weight, 0, 1) + final_bias

        if envswitch.who() == "shu":
            self.expander_nn = nn.Linear(self.hidden_size, self._tgt_vocab_size)
        else:
            # NOTE forced tying
            self.expander_nn = FinalLinear()

        self.label_smooth = LabelSmoothingKLDivLoss(0.1, self._tgt_vocab_size, 0)
        self.set_stepwise_training(False)

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # if self._fp16:
        #     self.half()

    def compute_length_predictor_loss(self, x_states, x_mask, y_mask):
        """Get the loss for length predictor.
        """
        y_lens = y_mask.sum(1)
        x_lens = x_mask.sum(1)
        delta = (y_lens - x_lens + 50.).long().clamp(0, 99)
        mean_z = (x_states * x_mask[:, :, None]).sum(1) / x_mask.sum(1)[:, None]
        logits = self.length_predictor(mean_z)
        length_loss = F.cross_entropy(logits, delta, reduction="mean")
        length_acc = self.to_float(logits.argmax(-1) == delta).mean()
        length_scores = {
            "len_loss": length_loss,
            "len_acc": length_acc
        }
        return length_scores

    def compute_vae_KL(self, p_prob, q_prob):
        # inputs : [batch_size, targets_length, latent_size]
        # outputs : [batch_size, targets_length, latent_size]
        q_mean, q_stddev = q_prob[..., :self.latent_dim], F.softplus(q_prob[..., self.latent_dim:])
        p_mean, p_stddev = p_prob[..., :self.latent_dim], F.softplus(p_prob[..., self.latent_dim:])

        kl = 0
        kl += torch.log(p_stddev + 1e-8) - torch.log(q_stddev + 1e-8)
        kl += (q_stddev ** 2 + (q_mean - p_mean)**2) / (2 * p_stddev**2)
        kl -= 0.5
        kl = kl.sum(-1)
        return kl

    def predict_length(self, p_states, x_mask, return_topk=1):
        """Predict the target length based on latent variables and source states.
        """
        mean_z = ( p_states * x_mask[:, :, None]).sum(1) / x_mask.sum(1)[:, None]
        logits = self.length_predictor(mean_z)
        delta = logits.argmax(-1) - 50
        if return_topk == 1:
            return delta
        else:
            topk = logits.topk(return_topk, dim=1).indices - 50
            return topk

    def compute_final_loss(self, q_prob, p_prob, y_mask, score_map):
        """ Compute the report the loss.
        """
        kl = self.compute_vae_KL(p_prob, q_prob) # [batch_size, targets_length, latent_size]
        # Apply budgets for KL divergence: KL = max(KL, budget)
        budget_upperbound = self.KL_budget
        if self.budget_annealing:
            step = OPTS.trainer.global_step()
            if OPTS.fastanneal:
                half_maxsteps = min(int(self.max_train_steps / 2), 50000) / 2
            else:
                half_maxsteps = float(self.max_train_steps / 2)
            if step > half_maxsteps:
                rate = (float(step) - half_maxsteps) / half_maxsteps
                min_budget = 0.
                budget = min_budget + (budget_upperbound - min_budget) * (1. - rate)
            else:
                budget = budget_upperbound
        else:
            budget = self.KL_budget
        score_map["KL_budget"] = torch.tensor(budget)
        # Compute KL divergence
        max_mask = self.to_float((kl - budget) > 0.)
        kl = kl * max_mask + (1. - max_mask) * budget
        kl_loss = (kl * y_mask / y_mask.shape[0]).sum()
        # Report KL divergence
        score_map["kl"] = kl_loss
        # Also report the averge KL for each token
        score_map["tok_kl"] = (kl * y_mask / y_mask.sum()).sum()
        # Report cross-entropy loss
        score_map["nll"] = score_map["loss"]
        # Cross-entropy loss is *already* backproped when computing softmaxes in shards
        # So only need to compute the remaining losses and then backprop them
        remain_loss = score_map["kl"].clone() * self.KL_weight
        if "len_loss" in score_map:
            remain_loss += score_map["len_loss"]
        # Report the combined loss
        score_map["loss"] = remain_loss + score_map["nll"]
        return score_map, remain_loss

    def compute_prior(self, y_mask, x_states, x_mask):
        y_shape = list(y_mask.shape)
        pos_states = self.pos_embed_layer(y_mask).expand(y_shape + [self.hidden_size])
        p_states = self.prior_encoder(pos_states, y_mask, x_states, x_mask)
        p_prob = self.p_hid2lat(p_states)
        return p_prob

    def compute_posterior(self, y, y_mask, x_states, x_mask):
        y_states = self.embed_layer(y)
        q_states = self.q_encoder_xy(y_states, y_mask, x_states, x_mask)
        q_prob = self.q_hid2lat(q_states)
        return q_prob

    def compute_logpy(self, logits, y, y_mask):
        shape = logits.shape
        nll = F.cross_entropy(
            logits.view(shape[0] * shape[1], -1),
            y.view(shape[0] * shape[1]), reduction="none", ignore_index=0)
        logpy = -1 * nll
        logpy = logpy.view(shape[0], shape[1])
        return logpy # [bsz, targets_length]

    def compute_logpz(self, z, y_mask, p_prob):
        latent_dim = self.latent_dim
        p_mean, p_stddev = p_prob[..., :latent_dim], F.softplus(p_prob[..., latent_dim:])
        logpz = -0.5 * ( (z - p_mean) / p_stddev ) ** 2 - torch.log(p_stddev * math.sqrt(2 * math.pi))
        logpz = logpz.sum(2)
        return logpz # [bsz, targets_length]

    def compute_logqz(self, z, y, y_mask, x_states, x_mask):
        latent_dim = self.latent_dim
        q_prob = self.compute_posterior(y, y_mask, x_states, x_mask)
        q_mean, q_stddev = q_prob[..., :latent_dim], F.softplus(q_prob[..., latent_dim:])
        logqz = -0.5 * ( (z - q_mean) / q_stddev ) ** 2 - torch.log(q_stddev * math.sqrt(2 * math.pi))
        logqz = logqz.sum(2)
        return logqz # [bsz, targets_length]

    def forward(self, x, y, sampling=False, return_code=False):
        """Model training.
        """
        score_map = {}
        x_mask = self.to_float(torch.ne(x, 0))
        y_mask = self.to_float(torch.ne(y, 0))
        batch_size = list(x.shape)[0]
        y_shape = list(y.shape)

        # Source sentence hidden states, shared between prior, posterior, decoder.
        x_states = self.embed_layer(x)
        x_states = self.x_encoder(x_states, x_mask)

        # Compute p(z|x)
        p_prob = self.compute_prior(y_mask, x_states, x_mask)

        # Compute q(z|x,y)
        q_prob = self.compute_posterior(y, y_mask, x_states, x_mask)
        q_mean, q_stddev = q_prob[..., :self.latent_dim], F.softplus(q_prob[..., self.latent_dim:])
        if not self.training:
            z_q = q_mean
        else:
            z_q = q_mean + q_stddev * torch.randn_like(q_stddev)

        # Compute length prediction loss
        length_scores = self.compute_length_predictor_loss(x_states, x_mask, y_mask)
        score_map.update(length_scores)

        # --------------------------  Decoder -------------------------------#
        hid_q = self.lat2hid(z_q)
        decoder_states = self.decoder(hid_q, y_mask, x_states, x_mask)

        # --------------------------  Compute losses ------------------------#
        decoder_outputs = TensorMap({"final_states": decoder_states})
        denom = x.shape[0]
        if self._shard_size is not None and self._shard_size > 0:
            loss_scores, decoder_tensors, decoder_grads = self.compute_shard_loss(
                decoder_outputs, y, y_mask, denominator=denom, ignore_first_token=False, backward=False
            )
            loss_scores["word_acc"] *= float(y_mask.shape[0]) / self.to_float(y_mask.sum())
            score_map.update(loss_scores)
        else:
            raise SystemError("Shard size must be setted or the memory is not enough for this model.")

        score_map, remain_loss = self.compute_final_loss(q_prob, p_prob, y_mask, score_map)
        # Report smoothed BLEU during validation
        if not torch.is_grad_enabled() and not self.training:
            decoder_outputs.unselect_batch()
            logits = self.expander_nn(decoder_outputs["final_states"])
            predictions = logits.argmax(-1)
            score_map["BLEU"] = - self.get_BLEU((predictions * y_mask).long(), (y * y_mask).long())

        # --------------------------  Bacprop gradient --------------------#
        if self._shard_size is not None and self._shard_size > 0 and decoder_tensors is not None:
            decoder_tensors.append(remain_loss)
            decoder_grads.append(None)
            torch.autograd.backward(decoder_tensors, decoder_grads)
        if torch.isnan(score_map["loss"]) or torch.isinf(score_map["loss"]):
            import pdb;pdb.set_trace()
        return score_map

    def compute_log_joint(self, x, z, y, y_mask):
        """Measure the ELBO in the inference time."""
        latent_dim = self.latent_dim
        x_mask = self.to_float(torch.ne(x, 0))
        y_length = y_mask.sum(1).float()
        batch_size = list(x.shape)[0]
        y_shape = list(y_mask.shape)
        vocab_size = self._tgt_vocab_size

        # Source sentence hidden states, shared between prior, posterior, decoder.
        x_states = self.embed_layer(x)
        x_states = self.x_encoder(x_states, x_mask)

        # Compute p(z|x)
        p_prob = self.compute_prior(y_mask, x_states, x_mask)

        logits = self.get_logits(z, y_mask, x_states, x_mask)
        y_pred = logits.argmax(-1)
        logpy = self.compute_logpy(logits, y, y_mask)
        logpz = self.compute_logpz(z, y_mask, p_prob)
        logpy = (logpy * y_mask).sum(1) / y_length
        logpz = (logpz * y_mask).sum(1) / y_length
        return logpy + logpz, logpy, logpz

    def compute_elbo(self, x, y, y_mask=None):
        """Measure the ELBO in the inference time."""
        latent_dim = self.latent_dim
        x_mask = self.to_float(torch.ne(x, 0))
        if y_mask == None:
            y_mask = self.to_float(torch.ne(y, 0))
        y_length = y_mask.sum(1).float()
        batch_size = list(x.shape)[0]
        y_shape = list(y.shape)
        vocab_size = self._tgt_vocab_size

        # Source sentence hidden states, shared between prior, posterior, decoder.
        x_states = self.embed_layer(x)
        x_states = self.x_encoder(x_states, x_mask)

        # Compute p(z|x)
        p_prob = self.compute_prior(y_mask, x_states, x_mask)

        # Compute q(z|x,y)
        q_prob = self.compute_posterior(y, y_mask, x_states, x_mask)
        q_mean, q_stddev = q_prob[..., :latent_dim], F.softplus(q_prob[..., latent_dim:])

        logpy_list, logpz_list, logqz_list = [], [], []
        for _ in range(20):
            z_q = q_mean + q_stddev * torch.randn_like(q_stddev)
            logits = self.get_logits(z_q, y_mask, x_states, x_mask)
            y_pred = logits.argmax(-1)
            logpy = self.compute_logpy(logits, y, y_mask)
            logpz = self.compute_logpz(z_q, y_mask, p_prob)
            logqz = self.compute_logqz(z_q, y_pred, y_mask, x_states, x_mask)
            logpy_list.append((logpy * y_mask).sum(1) / y_length)
            logpz_list.append((logpz * y_mask).sum(1) / y_length)
            logqz_list.append((logqz * y_mask).sum(1) / y_length)

        logpy_ = sum(logpy_list) / 20.0
        logpz_ = sum(logpz_list) / 20.0
        logqz_ = sum(logqz_list) / 20.0
        elbo = logpy_ + logpz_ - logqz_
        return elbo, logpy_, logpz_, logqz_

    def delta_refine(self, z, y_mask, x_states, x_mask, n_iter=1):
        for idx in range(n_iter):
            logits = self.get_logits(z, y_mask, x_states, x_mask)
            y_pred = logits.argmax(-1)
            y_pred = y_pred * y_mask.long()
            y_states = self.embed_layer(y_pred)
            q_states = self.q_encoder_xy(y_states, y_mask, x_states, x_mask)
            q_prob = self.q_hid2lat(q_states)
            z = q_prob[..., :self.latent_dim]
        return z

    def get_logits(self, z, y_mask, x_states, x_mask):
        hid = self.lat2hid(z)
        decoder_states = self.decoder(hid, y_mask, x_states, x_mask)
        logits = self.expander_nn(decoder_states)
        return logits

    def translate(self, x, refine_steps=0, y_mask=None):
        """ Testing codes.
        """
        x_mask = self.to_float(torch.ne(x, 0)).float()
        x_states = self.embed_layer(x)
        x_states = self.x_encoder(x_states, x_mask)

        # Predict length
        if y_mask is None:
            x_lens = x_mask.sum(1)
            delta = self.predict_length(x_states, x_mask)
            y_lens = delta.long() + x_lens.long()
            # y_lens = x_lens
            y_max_len = torch.max(y_lens.long()).item()
            batch_size = list(x_states.shape)[0]
            y_mask = torch.arange(y_max_len)[None, :].expand(batch_size, y_max_len)
            if torch.cuda.is_available():
                y_mask = y_mask.cuda()
            y_mask = (y_mask < y_lens[:, None]).float()

        # Compute p(z|x)
        pos_states = self.pos_embed_layer(y_mask[:, :, None]).expand(
          list(y_mask.shape) + [self.hidden_size])
        p_states = self.prior_encoder(pos_states, y_mask, x_states, x_mask)
        p_prob = self.p_hid2lat(p_states)
        z = p_prob[..., :self.latent_dim]

        # Perform refinement
        z = self.delta_refine(z, y_mask, x_states, x_mask, n_iter=refine_steps)

        hid = self.lat2hid(z)
        decoder_states = self.decoder(hid, y_mask, x_states, x_mask)
        logits = self.expander_nn(decoder_states)
        y_pred = logits.argmax(-1)
        y_pred = y_pred * y_mask.long()
        # y_pred = y_pred * x_mask.long()

        return y_pred, z, y_mask

    def get_BLEU(self, batch_y_hat, batch_y):
        """Get the average smoothed BLEU of the predictions."""
        hyps = batch_y_hat.tolist()
        refs = batch_y.tolist()
        bleus = []
        for hyp, ref in zip(hyps, refs):
            if 2 in hyp:
                hyp = hyp[:hyp.index(2)]
            if 2 in ref:
                ref = ref[:ref.index(2)]
            hyp = hyp[1:]
            ref = ref[1:]
            bleus.append(smoothed_bleu(hyp, ref))
        return torch.tensor(np.mean(bleus))
