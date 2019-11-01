# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import numpy as np
import torch


class TokenBlockDataset(torch.utils.data.Dataset):
    """Break a 1d tensor of tokens into blocks.

    The blocks are fetched from the original tensor so no additional memory is allocated.

    Args:
        tokens: 1d tensor of tokens to break into blocks
        sizes: sentence lengths (required for 'complete' and 'eos')
        block_size: maximum block size (ignored in 'eos' break mode)
        break_mode: Mode used for breaking tokens. Values can be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
            - 'complete': break tokens into blocks (up to block_size) such that
                blocks contains complete sentences, although block_size may be
                exceeded if some sentences exceed block_size
            - 'eos': each block contains one sentence (block_size is ignored)
        include_targets: return next tokens as targets
    """

    def __init__(self, tokens, sizes, block_size, break_mode=None, include_targets=False, ):
        super().__init__()

        self.tokens = tokens
        self.total_size = len(tokens)
        self.include_targets = include_targets
        self.slice_indices = []

        if break_mode is None or break_mode == 'none':
            length = math.ceil(len(tokens) / block_size)

            def block_at(i):
                start = i * block_size
                end = min(start + block_size, len(tokens))
                return (start, end)

            self.slice_indices = [block_at(i) for i in range(length)]
        elif break_mode == 'complete':
            assert sizes is not None and sum(sizes) == len(tokens), '{} != {}'.format(sum(sizes), len(tokens))
            tok_idx = 0
            sz_idx = 0
            curr_size = 0
            while sz_idx < len(sizes):
                if curr_size + sizes[sz_idx] <= block_size or curr_size == 0:
                    curr_size += sizes[sz_idx]
                    sz_idx += 1
                else:
                    self.slice_indices.append((tok_idx, tok_idx + curr_size))
                    tok_idx += curr_size
                    curr_size = 0
            if curr_size > 0:
                self.slice_indices.append((tok_idx, tok_idx + curr_size))
        elif break_mode == 'eos':
            assert sizes is not None and sum(sizes) == len(tokens), '{} != {}'.format(sum(sizes), len(tokens))
            curr = 0
            for sz in sizes:
                # skip samples with just 1 example (which would be just the eos token)
                if sz > 1:
                    self.slice_indices.append((curr, curr + sz))
                curr += sz
        else:
            raise ValueError('Invalid break_mode: ' + break_mode)

        self.sizes = np.array([e - s for s, e in self.slice_indices])

    def __getitem__(self, index):
        s, e = self.slice_indices[index]

        item = torch.LongTensor(self.tokens[s:e])

        if self.include_targets:
            # target is the sentence, for source, rotate item one token to the left (would start with eos)
            if s == 0:
                source = np.concatenate([self.tokens[-1:], self.tokens[0:e - 1]])
            else:
                source = self.tokens[s - 1:e - 1]

            return torch.LongTensor(source), item
        return item

    def __len__(self):
        return len(self.slice_indices)

# class TokenNERChunkBlockDataset(TokenBlockDataset):

#     def __init__(self, tokens, ner_tags, chunk_tags, sizes, block_size, 
#                     # pron_f, pron_m, pron_neut, pron_sg, pron_pl, pron_3p, pron_set, 
#                     break_mode=None, include_targets=False, 
#                     cbt_booktitle_idx=None):
#         super().__init__(tokens, sizes, block_size, break_mode, 
#                             include_targets, cbt_booktitle_idx)
#         self.ner_tags = ner_tags
#         self.chunk_tags = chunk_tags
#         # self.pron_f = pron_f
#         # self.pron_m = pron_m
#         # self.pron_neut = pron_neut
#         # self.pron_sg = pron_sg
#         # self.pron_pl = pron_pl
#         # self.pron_3p = pron_3p
#         # self.pron_set = pron_set

#     def __getitem__(self, index):
#         s, e = self.slice_indices[index]

#         token_item = torch.LongTensor(self.tokens[s:e])
#         ner_item = torch.LongTensor(self.ner_tags[s:e])
#         chunk_item = torch.LongTensor(self.chunk_tags[s:e])
#         # pron_f_item = torch.LongTensor(self.pron_f[s:e])
#         # pron_m_item = torch.LongTensor(self.pron_m[s:e])
#         # pron_neut_item = torch.LongTensor(self.pron_neut[s:e])
#         # pron_sg_item = torch.LongTensor(self.pron_sg[s:e])
#         # pron_pl_item = torch.LongTensor(self.pron_pl[s:e])
#         # pron_3p_item = torch.LongTensor(self.pron_3p[s:e])
#         # pron_set_item = torch.LongTensor(self.pron_set[s:e])

#         if self.include_targets:
#             # target is the sentence, for source, rotate item one token to the left (would start with eos)
#             if s == 0:
#                 source_token = np.concatenate([self.tokens[-1:], self.tokens[0:e - 1]])
#                 source_ner = np.concatenate([self.ner_tags[-1:], self.ner_tags[0:e - 1]])
#                 source_chunk = np.concatenate([self.chunk_tags[-1:], self.chunk_tags[0:e - 1]])
#                 # source_f = np.concatenate([self.pron_f[-1:], self.pron_f[0:e - 1]])
#                 # source_m = np.concatenate([self.pron_m[-1:], self.pron_m[0:e - 1]])
#                 # source_neut = np.concatenate([self.pron_neut[-1:], self.pron_neut[0:e - 1]])
#                 # source_sg = np.concatenate([self.pron_sg[-1:], self.pron_sg[0:e - 1]])
#                 # source_pl = np.concatenate([self.pron_pl[-1:], self.pron_pl[0:e - 1]])
#                 # source_3p = np.concatenate([self.pron_3p[-1:], self.pron_3p[0:e - 1]])
#                 # source_set = np.concatenate([self.pron_set[-1:], self.pron_set[0:e - 1]])
#             else:
#                 source_token = self.tokens[s - 1:e - 1]
#                 source_ner = self.ner_tags[s - 1:e - 1]
#                 source_chunk = self.chunk_tags[s - 1:e - 1]
#                 # source_f = self.pron_f[s - 1:e - 1]
#                 # source_m = self.pron_m[s - 1:e - 1]
#                 # source_neut = self.pron_neut[s - 1:e - 1]
#                 # source_sg = self.pron_sg[s - 1:e - 1]
#                 # source_pl = self.pron_pl[s - 1:e - 1]
#                 # source_3p = self.pron_3p[s - 1:e - 1]
#                 # source_set = self.pron_set[s - 1:e - 1]

#             return torch.LongTensor(source_token), torch.LongTensor(source_ner),\
#                     torch.LongTensor(source_chunk), token_item
#                     # torch.LongTensor(source_f),\
#                     # torch.LongTensor(source_m), torch.LongTensor(source_neut),\
#                     # torch.LongTensor(source_sg), torch.LongTensor(source_pl),\
#                     # torch.LongTensor(source_3p), torch.LongTensor(source_set),\
#         return token_item, ner_item, chunk_item
#                 # pron_m_item, pron_neut_item, pron_sg_item, pron_pl_item,\
#                 # pron_3p_item, pron_set_item

# class TokenNERChunkSupervisedBlockDataset(TokenBlockDataset):

#     def __init__(self, tokens, ner_tags, chunk_tags, gold_clusters, corefs, 
#                     sizes, block_size, break_mode=None, include_targets=False, 
#                     cbt_booktitle_idx=None):
#         super().__init__(tokens, sizes, block_size, break_mode, 
#                             include_targets, cbt_booktitle_idx)
#         assert tokens.shape == ner_tags.shape
#         assert tokens.shape == chunk_tags.shape
#         self.ner_tags = ner_tags
#         self.chunk_tags = chunk_tags
#         assert len(gold_clusters) == len(self.sizes)
#         self.gold_clusters = gold_clusters
#         assert len(corefs) == len(self.sizes)
#         self.corefs = corefs
#         self.stanford_gold_clusters = None


#     def __getitem__(self, index):
#         s, e = self.slice_indices[index]

#         token_item = torch.LongTensor(self.tokens[s:e])
#         ner_item = torch.LongTensor(self.ner_tags[s:e])
#         chunk_item = torch.LongTensor(self.chunk_tags[s:e])

#         if self.include_targets:
#             # target is the sentence, for source, rotate item one token to the left (would start with eos)
#             if s == 0:
#                 source_token = np.concatenate([self.tokens[-1:], self.tokens[0:e - 1]])
#                 source_ner = np.concatenate([self.ner_tags[-1:], self.ner_tags[0:e - 1]])
#                 source_chunk = np.concatenate([self.chunk_tags[-1:], self.chunk_tags[0:e - 1]])
#             else:
#                 source_token = self.tokens[s - 1:e - 1]
#                 source_ner = self.ner_tags[s - 1:e - 1]
#                 source_chunk = self.chunk_tags[s - 1:e - 1]
#             seqlen = len(source_token)
#             source_coref = np.zeros((seqlen, seqlen))
#             source_coref[1:, 1:] = self.corefs[index]
#             assert source_token[0] == 2 # pad
#             assert source_ner[0] == 0 # 0 for <eos>
#             assert source_chunk[0] == 0 # 0 for <eos>

#             return (
#                 torch.LongTensor(source_token), 
#                 torch.LongTensor(source_ner),
#                 torch.LongTensor(source_chunk), 
#                 torch.LongTensor(source_coref),
#                 # self.stanford_gold_clusters[index],
#                 self.gold_clusters[index],
#                 token_item,
#             )

#         return token_item, ner_item, chunk_item

# class TokenNERChunkSupervisedBlockGAPDataset(TokenBlockDataset):

#     def __init__(self, tokens, ner_tags, chunk_tags, sizes, block_size, 
#                     gap_ids, gap_pronouns, gap_pronouns_offsets,
#                     gap_a, gap_a_offsets, gap_a_coref,
#                     gap_b, gap_b_offsets, gap_b_coref,
#                     break_mode=None, include_targets=False, 
#                     cbt_booktitle_idx=None):
#         super().__init__(tokens, sizes, block_size, break_mode, 
#                             include_targets, cbt_booktitle_idx)
#         # assert tokens.shape == ner_tags.shape
#         # assert tokens.shape == chunk_tags.shape
#         self.ner_tags = ner_tags
#         self.chunk_tags = chunk_tags
#         # assert len(gold_clusters) == len(self.sizes)
#         # self.gold_clusters = gold_clusters
#         # assert len(corefs) == len(self.sizes)
#         # self.corefs = corefs
#         self.stanford_gold_clusters = None

#         self.gap_ids = gap_ids
#         self.gap_pronouns = gap_pronouns
#         self.gap_pronouns_offsets = gap_pronouns_offsets
#         self.gap_a = gap_a
#         self.gap_a_offsets = gap_a_offsets
#         self.gap_a_coref = gap_a_coref
#         self.gap_b = gap_b
#         self.gap_b_offsets = gap_b_offsets
#         self.gap_b_coref = gap_b_coref


#     def __getitem__(self, index):
#         s, e = self.slice_indices[index]

#         token_item = torch.LongTensor(self.tokens[s:e])
#         ner_item = torch.LongTensor(self.ner_tags[s:e])
#         # chunk_item = torch.LongTensor(self.chunk_tags[s:e])

#         if self.include_targets:
#             # target is the sentence, for source, rotate item one token to the left (would start with eos)
#             # if s == 0:
#             #     source_token = np.concatenate([self.tokens[-1:], self.tokens[0:e - 1]])
#             #     source_ner = np.concatenate([self.ner_tags[-1:], self.ner_tags[0:e - 1]])
#             #     source_chunk = np.concatenate([self.chunk_tags[-1:], self.chunk_tags[0:e - 1]])
#             # else:
#             #     source_token = self.tokens[s - 1:e - 1]
#             #     source_ner = self.ner_tags[s - 1:e - 1]
#             #     source_chunk = self.chunk_tags[s - 1:e - 1]
#             source_token = self.tokens[s:e]
#             source_ner = self.ner_tags[s:e]
#             # source_chunk = self.chunk_tags[s:e]
#             if index == len(self.slice_indices) - 1:
#                 assert e == len(self.tokens)
#                 token_item = torch.LongTensor(
#                     np.concatenate([self.tokens[s+1:e], self.tokens[0:1]])
#                 )
#             else:
#                 token_item = torch.LongTensor(self.tokens[s + 1:e + 1])
#             # TODO: improve the above so that the last element in token_item is eos
#             seqlen = len(source_token)
#             # source_coref = np.zeros((seqlen, seqlen))
#             # source_coref = self.corefs[index]
#             # # source_coref[1:, 1:] = self.corefs[index][:-1, :-1]
#             # # TODO: the above is clearly not correct, fix later!!!!
#             # # TODO: once fixed, uncomment the below
#             # # assert source_token[0] == 2 # pad
#             # # assert source_ner[0] == 0 # 0 for <eos>
#             # # assert source_chunk[0] == 0 # 0 for <eos>

#             return (
#                 torch.LongTensor(source_token), 
#                 torch.LongTensor(source_ner),
#                 # torch.LongTensor(source_chunk), 
#                 # torch.LongTensor(source_coref),
#                 # self.stanford_gold_clusters[index],
#                 # self.gold_clusters[index],
#                 self.gap_ids[index] if self.gap_ids is not None else None,
#                 self.gap_pronouns[index] if self.gap_pronouns is not None else None,
#                 self.gap_pronouns_offsets[index] if self.gap_pronouns_offsets is not None else None,
#                 self.gap_a[index] if self.gap_a is not None else None,
#                 self.gap_a_offsets[index] if self.gap_a_offsets is not None else None,
#                 self.gap_a_coref[index] if self.gap_a_coref is not None else None,
#                 self.gap_b[index] if self.gap_b is not None else None,
#                 self.gap_b_offsets[index] if self.gap_b_offsets is not None else None,
#                 self.gap_b_coref[index] if self.gap_b_coref is not None else None,
#                 token_item,
#             )

#         return token_item, ner_item, chunk_item


# class TokenNERChunkSupervisedBlockGAPGateDataset(TokenBlockDataset):

#     def __init__(self, tokens, ner_tags, chunk_tags, 
#                     overwrite_gates, update_gates,
#                     sizes, block_size, 
#                     gap_ids, gap_pronouns, gap_pronouns_offsets,
#                     gap_a, gap_a_offsets, gap_a_coref,
#                     gap_b, gap_b_offsets, gap_b_coref,
#                     gap_corefs, is_gap,
#                     break_mode=None, include_targets=False, 
#                     cbt_booktitle_idx=None):
#         super().__init__(tokens, sizes, block_size, break_mode, 
#                             include_targets, cbt_booktitle_idx)
#         # assert tokens.shape == ner_tags.shape
#         # assert tokens.shape == chunk_tags.shape
#         self.ner_tags = ner_tags
#         self.chunk_tags = chunk_tags

#         assert tokens.shape[0] == overwrite_gates.shape[0]
#         assert tokens.shape[0] == update_gates.shape[0]
#         self.overwrite_gates = overwrite_gates
#         self.update_gates = update_gates
#         # assert len(gold_clusters) == len(self.sizes)
#         # self.gold_clusters = gold_clusters
#         # assert len(corefs) == len(self.sizes)
#         # self.corefs = corefs
#         self.stanford_gold_clusters = None

#         self.gap_ids = gap_ids
#         self.gap_pronouns = gap_pronouns
#         self.gap_pronouns_offsets = gap_pronouns_offsets
#         self.gap_a = gap_a
#         self.gap_a_offsets = gap_a_offsets
#         self.gap_a_coref = gap_a_coref
#         self.gap_b = gap_b
#         self.gap_b_offsets = gap_b_offsets
#         self.gap_b_coref = gap_b_coref
#         self.gap_corefs = gap_corefs
#         self.is_gap = is_gap


#     def __getitem__(self, index):
#         s, e = self.slice_indices[index]

#         token_item = torch.LongTensor(self.tokens[s:e])
#         # ner_item = torch.LongTensor(self.ner_tags[s:e])
#         # chunk_item = torch.LongTensor(self.chunk_tags[s:e])

#         if self.include_targets:
#             # target is the sentence, for source, rotate item one token to the left (would start with eos)
#             if s == 0:
#                 source_token = np.concatenate([self.tokens[-1:], self.tokens[0:e - 1]])
#                 # source_ner = np.concatenate([self.ner_tags[-1:], self.ner_tags[0:e - 1]])
#                 # source_chunk = np.concatenate([self.chunk_tags[-1:], self.chunk_tags[0:e - 1]])
#                 source_overwrite_gates = np.concatenate([self.overwrite_gates[-1:], self.overwrite_gates[0:e - 1]])
#                 source_update_gates = np.concatenate([self.update_gates[-1:], self.update_gates[0:e - 1]])
#             else:
#                 source_token = self.tokens[s - 1:e - 1]
#                 # source_ner = self.ner_tags[s - 1:e - 1]
#                 # source_chunk = self.chunk_tags[s - 1:e - 1]
#                 source_overwrite_gates = self.overwrite_gates[s - 1:e - 1]
#                 source_update_gates = self.update_gates[s - 1:e - 1]
#             # source_token = self.tokens[s:e]
#             # source_ner = self.ner_tags[s:e]
#             # # source_chunk = self.chunk_tags[s:e]
#             # if index == len(self.slice_indices) - 1:
#             #     assert e == len(self.tokens)
#             #     token_item = torch.LongTensor(
#             #         np.concatenate([self.tokens[s+1:e], self.tokens[0:1]])
#             #     )
#             # else:
#             #     token_item = torch.LongTensor(self.tokens[s + 1:e + 1])
#             seqlen = len(source_token)
#             # source_coref = np.zeros((seqlen, seqlen))
#             # source_coref = self.corefs[index]
#             # # source_coref[1:, 1:] = self.corefs[index][:-1, :-1]
#             if self.is_gap[index] == 1.:
#                 assert source_token[0] == 2 # 2 for <eos>
#                 # assert source_ner[0] == 0 # 0 for <eos>
#                 # assert source_chunk[0] == 0 # 0 for <eos>
#                 assert source_overwrite_gates[0].sum() == 0 # 0 for <eos> (prev sent)
#                 assert source_update_gates[0].sum() == 0 # 0 for <eos> (prev sent)

#             def _increase_offsets(offsets):
#                 return (offsets[0] + 1, offsets[1] + 1)

#             def _increase_offsets_corefs(corefs):
#                 seqlen = corefs.shape[0]
#                 ret = np.zeros((seqlen, seqlen))
#                 ret[1:, 1:] = corefs[:-1, :-1]
#                 return ret

#             return (
#                 torch.LongTensor(source_token), 
#                 # torch.LongTensor(source_ner),
#                 # torch.LongTensor(source_chunk), 
#                 torch.LongTensor(source_overwrite_gates),
#                 torch.LongTensor(source_update_gates),
#                 # torch.LongTensor(source_coref),
#                 # self.stanford_gold_clusters[index],
#                 # self.gold_clusters[index],
#                 self.gap_ids[index], # if self.gap_ids is not None else None,
#                 self.gap_pronouns[index], # if self.gap_pronouns is not None else None,
#                 _increase_offsets(self.gap_pronouns_offsets[index]), # if self.gap_pronouns_offsets is not None else None,
#                 self.gap_a[index], # if self.gap_a is not None else None,
#                 _increase_offsets(self.gap_a_offsets[index]), # if self.gap_a_offsets is not None else None,
#                 self.gap_a_coref[index], # if self.gap_a_coref is not None else None,
#                 self.gap_b[index], # if self.gap_b is not None else None,
#                 _increase_offsets(self.gap_b_offsets[index]), # if self.gap_b_offsets is not None else None,
#                 self.gap_b_coref[index], # if self.gap_b_coref is not None else None,
#                 torch.FloatTensor(_increase_offsets_corefs(self.gap_corefs[index])),
#                 self.is_gap[index],
#                 token_item,
#             )

#         return token_item, ner_item, chunk_item