#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append(".")

from lanmt.lib_lanmt_modules import TransformerEncoder
from lanmt.lib_lanmt_model import LANMTModel
from lanmt.lib_simple_encoders import ConvolutionalEncoder
from lanmt.lib_vae import VAEBottleneck
from nmtlab.models import Transformer
from nmtlab.utils import OPTS


class LatentEncodingNetwork(Transformer):

    def __init__(self, prior_layers=0, q_layers=3, decoder_layers=1,
                 latent_dim=8,
                 KL_budget=1., KL_weight=1.,
                 budget_annealing=True,
                 max_train_steps=100000,
                 **kwargs):
        self.prior_layers = prior_layers
        self.decoder_layers = decoder_layers
        self.q_layers = prior_layers
        self.latent_dim = latent_dim
        self.KL_budget = KL_budget
        self.KL_weight = KL_weight
        self.budget_annealing = budget_annealing
        self.max_train_steps = max_train_steps
        self.training_criteria = "loss"
        OPTS.fixbug1 = True
        OPTS.fixbug2 = True
        super(LatentEncodingNetwork, self).__init__(**kwargs)

    def prepare(self):
        """Define the modules
        """
        # Embedding layers
        self.embed_layer = TransformerEmbedding(self._src_vocab_size, self.embed_size)
        self.pos_embed_layer = PositionalEmbedding(self.hidden_size)
        # Prior p(z|x)
        # Approximator q(z|x)
        self.q_encoder = ConvolutionalEncoder(self.embed_layer, self.hidden_size, self.q_layers)
        # Decoder p(x|z)
        self.decoder = ConvolutionalEncoder(None, self.hidden_size, self.decoder_layers, skip_connect=True)
        # Bottleneck
        self.bottleneck = VAEBottleneck(self.hidden_size, z_size=self.latent_dim, standard_var=True)
        self.latent2vector_nn = nn.Linear(self.latent_dim, self.hidden_size)
        # Word probability estimator
        self.expander_nn = nn.Linear(self.hidden_size, self._tgt_vocab_size)
        self.label_smooth = LabelSmoothingKLDivLoss(0.1, self._tgt_vocab_size, 0)
        self.set_stepwise_training(False)

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_Q(self, seq):
        mask = self.to_float(torch.ne(seq, 0))
        q_states = self.compute_Q_states(seq, mask)
        sampled_latent, q_prob = self.sample_from_Q(q_states, sampling=False)
        return sampled_latent, q_prob

    def compute_Q_states(self, seq, mask):
        states = self.q_encoder(seq, mask)
        return states

    def sample_from_Q(self, q_states, sampling=True):
        sampled_z, q_prob = self.bottleneck(q_states, sampling=sampling)
        full_vector = self.latent2vector_nn(sampled_z)
        return full_vector, q_prob

    def compute_vae_KL(self, prior_prob, q_prob):
        """Compute KL divergence given two Gaussians.
        """
        mu1 = q_prob[:, :, :self.latent_dim]
        var1 = F.softplus(q_prob[:, :, self.latent_dim:])
        mu2 = prior_prob[:, :, :self.latent_dim]
        var2 = F.softplus(prior_prob[:, :, self.latent_dim:])
        kl = torch.log(var2 / (var1 + 1e-8) + 1e-8) + (
                    (torch.pow(var1, 2) + torch.pow(mu1 - mu2, 2)) / (2 * torch.pow(var2, 2))) - 0.5
        kl = kl.sum(-1)
        return kl

    def deterministic_sample_from_prob(self, z_prob):
        """ Obtain the mean vectors from multi-variate normal distributions.
        """
        mean_vector = z_prob[:, :, :self.latent_dim]
        full_vector = self.latent2vector_nn(mean_vector)
        return full_vector

    def compute_final_loss(self, q_prob, prior_prob, x_mask, score_map):
        """ Compute the report the loss.
        """
        kl = self.compute_vae_KL(prior_prob, q_prob)
        # Apply budgets for KL divergence: KL = max(KL, budget)
        budget_upperbound = self.KL_budget
        if self.budget_annealing:
            step = OPTS.trainer.global_step()
            if OPTS.beginanneal < 0:
                beginstep = float(self.max_train_steps / 2)
            else:
                beginstep = float(OPTS.beginanneal)
            if step > beginstep:
                max_train_steps = min(int(self.max_train_steps/2), 50000) if OPTS.fastanneal else self.max_train_steps
                rate = (float(step) - beginstep) / (max_train_steps - beginstep)
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
        kl_loss = (kl * x_mask / x_mask.shape[0]).sum()
        if OPTS.nokl:
            kl_loss *= 0.0000001
        # Report KL divergence
        score_map["kl"] = kl_loss
        # Also report the averge KL for each token
        score_map["tok_kl"] = (kl * x_mask / x_mask.sum()).sum()
        # Report cross-entropy loss
        score_map["nll"] = score_map["loss"]
        # Cross-entropy loss is *already* backproped when computing softmaxes in shards
        # So only need to compute the remaining losses and then backprop them
        remain_loss = score_map["kl"].clone() * self.KL_weight
        remain_loss = score_map["kl"].clone() * self.KL_weight
        if "len_loss" in score_map:
            remain_loss = remain_loss + score_map["len_loss"]
        # Report the combined loss
        score_map["loss"] = remain_loss + score_map["nll"]
        return score_map, remain_loss

    def forward(self, x, y, sampling=False, return_code=False):
        """Model training.
        """
        score_map = {}
        seq = y
        mask = self.to_float(torch.ne(seq, 0))

        # ----------- Compute prior and approximated posterior -------------#
        # Compute p(z|x)
        prior_prob = self.standard_gaussian_dist(seq.shape[0], seq.shape[1])
        # Compute q(z|x,y) and sample z
        q_states = self.compute_Q_states(seq, mask)
        # Sample latent variables from q(z|x,y)
        sampled_z, q_prob = self.sample_from_Q(q_states)

        # --------------------------  Decoder -------------------------------#
        decoder_states = self.decoder(sampled_z, mask)

        # --------------------------  Compute losses ------------------------#
        decoder_outputs = TensorMap({"final_states": decoder_states})
        denom = seq.shape[0]
        if self._shard_size is not None and self._shard_size > 0:
            loss_scores, decoder_tensors, decoder_grads = self.compute_shard_loss(
                decoder_outputs, seq, mask, denominator=denom, ignore_first_token=False, backward=False
            )
            loss_scores["word_acc"] *= float(mask.shape[0]) / self.to_float(mask.sum())
            score_map.update(loss_scores)
        else:
            raise SystemError("Shard size must be setted or the memory is not enough for this model.")

        score_map, remain_loss = self.compute_final_loss(q_prob, prior_prob, mask, score_map)
        # --------------------------  Bacprop gradient --------------------#
        if self._shard_size is not None and self._shard_size > 0 and decoder_tensors is not None:
            decoder_tensors.append(remain_loss)
            decoder_grads.append(None)
            torch.autograd.backward(decoder_tensors, decoder_grads)
        return score_map

    def translate(self, x, latent=None, prior_states=None, refine_step=0):
        """ Testing codes.
        """
        x_mask = self.to_float(torch.ne(x, 0))
        # Compute p(z|x)
        if prior_states is None:
            prior_states = self.prior_encoder(x, x_mask)
        # Sample latent variables from prior if it's not given
        if latent is None:
            if OPTS.zeroprior:
                prior_prob = self.standard_gaussian_dist(x.shape[0], x.shape[1])
            else:
                prior_prob = self.prior_prob_estimator(prior_states)
            if not OPTS.Tlatent_search:
                if OPTS.denoise:
                    z = prior_prob[:, :, :self.latent_dim]
                    prior_prob = OPTS.denoise.refine(z, self.x_embed_layer(x), x_mask)
                latent = self.deterministic_sample_from_prob(prior_prob)
            else:
                latent = self.bottleneck.sample_any_dist(prior_prob, samples=OPTS.Tcandidate_num, noise_level=0.5)
                latent = self.latent2vector_nn(latent)
        # Predict length
        length_delta = self.predict_length(prior_states, latent, x_mask)
        # Adjust the number of latent
        converted_z, y_mask, y_lens = self.convert_length_with_delta(latent, x_mask, length_delta + 1)
        if converted_z.size(1) == 0:
            return None, latent, prior_prob.argmax(-1)
        # Run decoder to predict the target words
        decoder_states = self.decoder(converted_z, y_mask, prior_states, x_mask)
        logits = self.expander_nn(decoder_states)
        # Get the target predictions
        if OPTS.Tlatent_search and not OPTS.Tteacher_rescore:
            # Latent search without teacher rescoring is dangeous
            # because the generative model framework can't effeciently and correctly score hypotheses
            if refine_step == OPTS.Trefine_steps:
                # In the finally step, pick the best hypotheses
                logprobs, preds = torch.log_softmax(logits, 2).max(2)
                logprobs = (logprobs * y_mask).sum(1)  # x batch x 1
                preds = preds * y_mask.long()
                # after deterimistic refinement
                pred = preds[logprobs.argmax()].unsqueeze(0)
            else:
                # Just return all candidates
                pred = logits.argmax(-1)
                pred = pred * y_mask.long()
        else:
            pred = logits.argmax(-1)

        return pred, latent, prior_states

    def standard_gaussian_dist(self, batch_size, seq_size):
        shape = (batch_size, seq_size, self.latent_dim)
        return torch.cat([torch.zeros(shape).cuda(), torch.ones(shape).cuda() * 0.55], 2)

