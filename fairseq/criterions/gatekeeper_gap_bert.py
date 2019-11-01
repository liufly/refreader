# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import torch.nn.functional as F
import torch
import sys
import os

from fairseq import utils
from . import FairseqCriterion, register_criterion
from fairseq.models import GapBertDecoder, FairseqLanguageModel
from fairseq.models.gap_evaluator import GAPEvaluator

import math
import numpy as np
from collections import defaultdict
from gap_scorer import Annotation
from constants import Gender, PRONOUNS

@register_criterion('gatekeeper_gap_bert')
class GateKeeperGapBertCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.coref_bce_loss = torch.nn.BCELoss(reduction='none')
        self.coref_evaluator = GAPEvaluator()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--threshold', default=None, type=float, metavar='D',
                            help='coref threshold')
        parser.add_argument('--self-link-loss-weight', default=0.1, type=float,
                            help='self link loss weight')
        parser.add_argument('--coref-link-loss-weight', default=5.0, type=float,
                            help='coreferential link loss weight')
        parser.add_argument('--coref-link-positive-loss-weight', default=1.0, type=float,
                            help='coreferential link positive loss weight')
        parser.add_argument('--coref-link-negative-loss-weight', default=10.0, type=float,
                            help='coreferential link negative loss weight')
        parser.add_argument('--non-link-loss-weight', default=1.0, type=float,
                            help='non link loss weight')

    def forward(self, model, sample, reduce=True):
        if not isinstance(model, FairseqLanguageModel):
            raise ValueError("can only use gatekeepercriterion with models that inherit from FairseqLanguageModelSpan")

        if not isinstance(model.decoder, GapBertDecoder):
            raise ValueError("can only use gatekeepercriterion with decoders that inherit from SoftSupervisedDecoder")
        
        if 'epoch' in sample:
            epoch = sample['epoch']
            model.decoder.set_epoch(epoch)

        gap_data = sample['gap_data']
        gap_corefs = sample['gap_corefs']
        gap_corefs_masks = sample['gap_corefs_mask']
        gap_bert_weights = sample['gap_bert_weights']

        src_tokens = sample['net_input']['src_tokens']
        assert len(gap_data) == src_tokens.size(0)

        net_output = model(**{"src_tokens": gap_bert_weights})

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        xent_loss = F.nll_loss(
            lprobs, target, size_average=False, ignore_index=self.padding_idx,
            reduce=reduce
        )
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        
        rnn_outs, corefs = net_output

        seqlen = gap_corefs.size(1)
        bsz = gap_corefs.size(0)

        ones = model.decoder.new_ones(1)
        zeros = model.decoder.new_zeros(1)

        # all coref == True mask
        corefs_gap_coreference_positive_mask = torch.where(
            gap_corefs == 1., ones, zeros
        )
        # plus all coref == False mask
        corefs_gap_coreference_negative_mask = torch.where(
            gap_corefs == 2., ones, zeros
        )
        corefs_gap_coreferential_mask = \
            corefs_gap_coreference_positive_mask \
                + corefs_gap_coreference_negative_mask
        
        # coref self-link mask
        corefs_gap_selflink_mask = torch.where(gap_corefs == 3., ones, zeros)

        gap_corefs = torch.where(gap_corefs == 2., zeros, gap_corefs)
        gap_corefs = torch.where(gap_corefs == 3., ones, gap_corefs)

        triu_mask = model.decoder.new_ones(seqlen, seqlen)
        triu_mask = torch.triu(triu_mask, diagonal=1)
        triu_mask = triu_mask.view(1, seqlen, seqlen)

        gap_corefs = gap_corefs.type_as(corefs)
        gap_corefs_masks = gap_corefs_masks.type_as(corefs) * triu_mask
        coref_loss = self.coref_bce_loss(
            torch.clamp(corefs, min=0.0, max=1.0),
            torch.clamp(gap_corefs, min=0.0, max=1.0),
        ) * gap_corefs_masks
        # [bsz, seqlen, seqlen]

        corefs_gap_non_mask = \
            (1. - (corefs_gap_coreferential_mask + corefs_gap_selflink_mask)) \
                * gap_corefs_masks

        corefs_gap_selflink_loss = coref_loss * corefs_gap_selflink_mask
        # [bsz, seqlen, seqlen]

        corefs_gap_coreference_positive_loss = \
            coref_loss * corefs_gap_coreference_positive_mask \
                * self.args.coref_link_positive_loss_weight
        corefs_gap_coreference_negative_loss = \
            coref_loss * corefs_gap_coreference_negative_mask \
                * self.args.coref_link_negative_loss_weight
        corefs_gap_coreference_loss = \
            corefs_gap_coreference_positive_loss \
                + corefs_gap_coreference_negative_loss
        # [bsz, seqlen, seqlen]

        corefs_gap_non_loss = coref_loss * corefs_gap_non_mask \
                            * self.args.non_link_loss_weight
        # [bsz, seqlen, seqlen]

        corefs_gap_coreference_loss_weight = \
            (self.args.coref_link_loss_weight * ones).view(-1, 1, 1)
        # [bsz, seqlen, seqlen]
        corefs_gap_selflink_loss_weight = \
            (self.args.self_link_loss_weight * ones).view(-1, 1, 1)
        # [bsz, seqlen, seqlen]
        corefs_gap_non_loss_weight = \
            (self.args.non_link_loss_weight * ones).view(-1, 1, 1)
        # [bsz, seqlen, seqlen]

        corefs_gap_selflink_loss = \
            corefs_gap_selflink_loss * corefs_gap_selflink_loss_weight
        # [bsz, seqlen, seqlen]
        corefs_gap_coreference_loss = \
            corefs_gap_coreference_loss * corefs_gap_coreference_loss_weight
        # [bsz, seqlen, seqlen]
        corefs_gap_non_loss = \
            corefs_gap_non_loss * corefs_gap_non_loss_weight
        # [bsz, seqlen, seqlen]

        coref_loss = \
            corefs_gap_selflink_loss.sum() \
                + corefs_gap_coreference_loss.sum() \
                + corefs_gap_non_loss.sum()

        if model.training:
            loss = coref_loss + xent_loss * 0
        else:
            loss = xent_loss

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }

        corefs_symmetric = corefs + corefs.transpose(1, 2)
        corefs_eye = corefs_symmetric \
            + torch.eye(seqlen).unsqueeze(0).type_as(corefs_symmetric)
        corefs_eye = corefs_eye[:, :, :].detach().cpu()
        
        corefs_eye = corefs_eye.numpy()
        assert len(corefs_eye) == len(gap_data)

        # evaluation based on threshold
        predicted_thresholds = {}
        threshold_range = np.arange(0.0, 1.0, 0.01) 
        if self.args.threshold is not None:
            threshold_range = [self.args.threshold]
        if not model.training:
            for threshold in threshold_range:
                predicted_annotations = defaultdict(Annotation)
                for gap_instance, coref in zip(gap_data, corefs_eye):
                    g_id = gap_instance.example_id
                    pronoun = gap_instance.pronoun
                    pronoun_offset = (
                        gap_instance.pronoun_offset_start, 
                        gap_instance.pronoun_offset_end
                    )
                    a_offset = (
                        gap_instance.a_offset_start, gap_instance.a_offset_end
                    )
                    b_offset = (
                        gap_instance.b_offset_start, gap_instance.b_offset_end
                    )
                    assert pronoun_offset[0] == pronoun_offset[1]
                    annotation = self.make_annotation(pronoun, False, False)
                    for col in range(a_offset[0], a_offset[1] + 1):
                        if coref[pronoun_offset[0], col] >= threshold:
                            annotation.name_a_coref = True
                            break
                    for col in range(b_offset[0], b_offset[1] + 1):
                        if coref[pronoun_offset[0], col] >= threshold:
                            annotation.name_b_coref = True
                            break
                    
                    predicted_annotations[g_id] = annotation
                assert threshold not in predicted_thresholds
                predicted_thresholds[threshold] = predicted_annotations

        gold_annotations = defaultdict(Annotation)
        for gap_instance in gap_data:
            gold_annotations[gap_instance.example_id] = \
                self.make_annotation(
                    gap_instance.pronoun, 
                    gap_instance.a_coref, 
                    gap_instance.b_coref
                )

        if not model.training:
            logging_output['predicted_results'] = predicted_thresholds
            logging_output['gold_clusters'] = gold_annotations
        
        return loss, sample_size, logging_output
    
    def make_annotation(self, pronoun, a_coref, b_coref):
        annotation = Annotation()
        annotation.name_a_coref = a_coref
        annotation.name_b_coref = b_coref
        gender = PRONOUNS.get(pronoun.lower(), Gender.UNKNOWN)
        assert gender != Gender.UNKNOWN
        annotation.gender = gender
        return annotation

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'sample_size': sample_size,
        }

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        predicted_results, gold_clusters = [], []
        for log in logging_outputs:
            if 'predicted_results' in log and 'gold_clusters' in log:
                assert len(logging_outputs) == 1
                predicted_results = log['predicted_results']
                gold_clusters = log['gold_clusters']

        agg_output['predicted_results'] = predicted_results
        agg_output['gold_clusters'] = gold_clusters
        return agg_output
