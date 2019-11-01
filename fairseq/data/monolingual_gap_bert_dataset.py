# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset
from fairseq.data.gap_reader import GAP_Record


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples], pad_idx, eos_idx, left_pad=False,
        )
    
    gap_corefs, gap_corefs_mask = data_utils.collate_corefs(
        [s['gap_corefs'] for s in samples], empty_idx=0,
    )

    gap_data = [s['gap_data'] for s in samples]

    actual_lens = [len(s['source_token']) for s in samples]

    gap_bert_weights = data_utils.collate_bert_weights(
        [s['gap_bert_weights'] for s in samples], empty_idx=0,
    )

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'ntokens': sum(len(s['target']) for s in samples),
        'net_input': {
            'src_tokens': merge('source_token'),
        },
        'target': merge('target'),
        'gap_corefs': gap_corefs,
        'gap_corefs_mask': gap_corefs_mask,
        'actual_lens': actual_lens,
        'gap_data': gap_data,
        'gap_bert_weights': gap_bert_weights,
    }


class MonolingualGapBertDataset(FairseqDataset):
    """A wrapper around torch.utils.data.Dataset for monolingual data."""

    def __init__(self, dataset, sizes, token_dictionary, shuffle):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.token_dictionary = token_dictionary
        self.shuffle = shuffle

    def __getitem__(self, index):
        source_token, gap_data, gap_corefs, gap_bert_weights, target = self.dataset[index]
        return {
            'id': index, 
            'source_token': source_token, 
            'gap_data': gap_data,
            'gap_corefs': gap_corefs,
            'gap_bert_weights': gap_bert_weights,
            'target': target
        }

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(
            samples, self.token_dictionary.pad(), self.token_dictionary.eos()
        )

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        assert isinstance(max_positions, float) or isinstance(max_positions, int)
        tgt_len = min(tgt_len, max_positions)
        bsz = num_tokens // tgt_len
        target = self.token_dictionary.dummy_sentence(tgt_len + 1)
        source, target = target[:-1], target[1:]
        source_gap_data = GAP_Record(
            "", "", "she", 0, 0, "", 0, 0, False, "", 0, 0, False
        )
        source_gap_corefs = torch.Tensor(np.zeros((tgt_len, tgt_len))).float()
        source_gap_corefs_mask = torch.Tensor(np.zeros((tgt_len, tgt_len))).float()
        source_gap_bert_weights = torch.Tensor(np.zeros((4, tgt_len, 768))).float()
        return self.collater([
            {'id': i, 'source_token': source,
            'gap_data': source_gap_data,
            'gap_corefs': source_gap_corefs,
            'gap_corefs_mask': source_gap_corefs_mask,
            'gap_bert_weights': source_gap_bert_weights,
            'target': target}
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return self.sizes[index]

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(np.flip(self.sizes, 0))
        return np.lexsort(order)

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        assert isinstance(max_positions, float) or isinstance(max_positions, int)
        return self.sizes[index] <= max_positions
