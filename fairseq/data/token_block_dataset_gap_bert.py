# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import numpy as np
import torch
from fairseq.data.token_block_dataset import TokenBlockDataset
from fairseq.data.gap_reader import GAP_Record

class TokenBlockGapBertDataset(TokenBlockDataset):

    def __init__(self, tokens, sizes, block_size, gap_data, gap_corefs, 
                    gap_bert_weights, break_mode=None, include_targets=False):
        super().__init__(tokens, sizes, block_size, break_mode, 
                            include_targets)

        self.gap_data = gap_data
        self.gap_corefs = gap_corefs
        self.gap_bert_weights = gap_bert_weights


    def __getitem__(self, index):
        s, e = self.slice_indices[index]

        token_item = torch.LongTensor(self.tokens[s:e])

        assert self.include_targets == True

        if s == 0:
            source_token = np.concatenate([self.tokens[-1:], self.tokens[0:e - 1]])
        else:
            source_token = self.tokens[s - 1:e - 1]
        seqlen = len(source_token)
        assert source_token[0] == 2 # 2 for <eos>

        def _increase_offsets(offsets):
            return (offsets[0] + 1, offsets[1] + 1)

        def _increase_offsets_corefs(corefs):
            seqlen = corefs.shape[0]
            ret = np.zeros((seqlen, seqlen))
            ret[1:, 1:] = corefs[:-1, :-1]
            return ret
        
        def _increase_offsets(data):
            return GAP_Record(
                data.example_id,
                data.text,
                data.pronoun,
                data.pronoun_offset_start + 1,
                data.pronoun_offset_end + 1,
                data.a,
                data.a_offset_start + 1,
                data.a_offset_end + 1,
                data.a_coref,
                data.b,
                data.b_offset_start + 1,
                data.b_offset_end + 1,
                data.b_coref
            )
        
        def _increase_offsets_bert(bert_weights):
            bert_weights_shape = bert_weights.shape
            eos_padding = np.zeros((bert_weights_shape[0], 1, bert_weights_shape[2]))
            return np.concatenate([eos_padding, bert_weights], axis=1)

        return (
            torch.LongTensor(source_token), 
            _increase_offsets(self.gap_data[index]),
            torch.FloatTensor(_increase_offsets_corefs(self.gap_corefs[index])),
            torch.FloatTensor(_increase_offsets_bert(self.gap_bert_weights[index])),
            token_item,
        )