# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os
import sys
import random
import pickle

from torch.utils.data import ConcatDataset

from fairseq.data import (
    Dictionary, IndexedInMemoryDataset, IndexedRawTextDataset,
    MonolingualGapBertDataset, TokenBlockGapBertDataset,
    GAP_Reader, Bert_Reader
)

from . import FairseqTask, register_task
from fairseq.models.pronouns import PronounLexicon

@register_task('language_modeling_refreader_gap_bert')
class LanguageModelingRefreaderGapBertTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--sample-break-mode', metavar='VAL', default='eos',
                            choices=['eos'],
                            help='If set to "eos" (the only supported mode),'
                                 'includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=128, type=int, 
                            metavar='N',
                            help='max number of tokens per sample for LM dataset')


    def __init__(self, args, token_dictionary, ner_dictionary):
        super().__init__(args)
        self.token_dictionary = token_dictionary
        self.ner_dictionary = ner_dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        token_dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
        print('| token dictionary: {} types'.format(len(token_dictionary)))
        ner_dictionary = None
        return cls(args, token_dictionary, ner_dictionary)

    def generate_gap_coref_supervision(self, gap_data, seqlens):
        nb_instances = len(gap_data)

        coref_supervisions = []

        for i in range(nb_instances):
            seqlen = seqlens[i]
            data = gap_data[i]
            
            pronoun_offset = (
                data.pronoun_offset_start,
                data.pronoun_offset_end
            )
            a_offset = (data.a_offset_start, data.a_offset_end)
            b_offset = (data.b_offset_start, data.b_offset_end)
            a_coref = data.a_coref
            b_coref = data.b_coref
            coref_supervision = np.zeros((seqlen, seqlen))

            # test non-overlapping
            assert pronoun_offset[0] == pronoun_offset[1]
            assert pronoun_offset[0] < a_offset[0] \
                    or pronoun_offset[0] > a_offset[1]
            assert pronoun_offset[0] < b_offset[0] \
                    or pronoun_offset[0] > b_offset[1]
            assert b_offset[0] <= b_offset[1]
            assert a_offset[1] < b_offset[0]

            pronoun_offset = pronoun_offset[0]

            assert (a_coref and (not b_coref)) \
                    or ((not a_coref) and b_coref) \
                    or ((not a_coref) and (not b_coref))
            
            a_coref_value = 1 if a_coref else 2
            # if a_coref:
            if a_offset[1] < pronoun_offset:
                coref_supervision[a_offset[0]:a_offset[1]+1, pronoun_offset] = a_coref_value
            elif pronoun_offset < a_offset[0]:
                coref_supervision[pronoun_offset, a_offset[0]:a_offset[1]+1] = a_coref_value
            else:
                assert False # should never happen

            b_coref_value = 1 if b_coref else 2
            # if b_coref:
            if b_offset[1] < pronoun_offset:
                coref_supervision[b_offset[0]:b_offset[1]+1, pronoun_offset] = b_coref_value
            elif pronoun_offset < b_offset[0]:
                coref_supervision[pronoun_offset, b_offset[0]:b_offset[1]+1] = b_coref_value
            else:
                assert False # should never happen
            
            def add_self_links(coref_mat, offset_start, offset_end):
                for row_idx in range(offset_start, offset_end):
                    coref_mat[row_idx, row_idx+1:offset_end] = 3
                return coref_mat

            coref_supervision = add_self_links(
                coref_supervision, a_offset[0], a_offset[1] + 1
            )
            coref_supervision = add_self_links(
                coref_supervision, b_offset[0], b_offset[1] + 1
            )

            coref_supervisions.append(coref_supervision)
        
        return coref_supervisions

    def load_dataset(self, split, combine=False):
        """Load a dataset split."""

        loaded_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            token_path = os.path.join(self.args.data, split_k)

            if IndexedInMemoryDataset.exists(token_path):
                token_ds = IndexedInMemoryDataset(
                    token_path, fix_lua_indexing=True
                )
                tokens = token_ds.buffer

                sizes = token_ds.sizes

                in_tsv_file_path = os.path.join(
                    self.args.data, f'gap-{split}.bert.tsv'
                )
                gap_reader = GAP_Reader(in_tsv_file_path, is_gold=True)
                gap_data = gap_reader.read()

                in_bert_file_path = os.path.join(
                    self.args.data, f'gap-{split}.bert.jsonl'
                )

                gap_bert_reader = Bert_Reader(in_bert_file_path)
                gap_bert_data = gap_bert_reader.read()
                gap_bert_weights = [
                    bert_weights for _, bert_weights in gap_bert_data
                ]
                
                

                gap_texts = [d.text.split() for d in gap_data]
                assert np.array_equal(
                    sizes, 
                    [len(t) + 1 for t in gap_texts]
                )
                assert np.array_equal(
                    sizes, 
                    [len(bert_tokens) + 1 for bert_tokens, _ in gap_bert_data]
                )
                assert np.array_equal(
                    [d.text.split(" ") for d in gap_data],
                    [bert_tokens for bert_tokens, _ in gap_bert_data]
                )

                gap_corefs = self.generate_gap_coref_supervision(
                    gap_data, sizes
                )
                assert len(gap_data) == len(gap_corefs)

            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

            loaded_datasets.append(
                TokenBlockGapBertDataset(
                    tokens, sizes, self.args.tokens_per_sample,
                    gap_data, gap_corefs, gap_bert_weights,
                    break_mode=self.args.sample_break_mode,
                    include_targets=True
                )
            )

            if split == "train":
                gap_dataset = TokenBlockGapBertDataset(
                    tokens, sizes, self.args.tokens_per_sample,
                    gap_data, gap_corefs, gap_bert_weights,
                    self.args.sample_break_mode,
                    include_targets=True
                )
                self.datasets["train_gap_only"] = MonolingualGapBertDataset(
                    gap_dataset, gap_dataset.sizes, self.token_dictionary, shuffle=False
                )

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[-1])))

            if not combine:
                break

        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
            sizes = dataset.sizes
        else:
            dataset = ConcatDataset(loaded_datasets)
            sizes = np.concatenate([ds.sizes for ds in loaded_datasets])

        self.datasets[split] = MonolingualGapBertDataset(
            dataset, sizes, self.token_dictionary, shuffle=False
        )

    @property
    def source_dictionary(self):
        return self.token_dictionary

    @property
    def target_dictionary(self):
        return self.token_dictionary
