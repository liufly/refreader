# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import LongTensor, FloatTensor
import sys
import numpy as np
import copy

from fairseq import options, utils

from . import (
    FairseqDecoder, GapBertDecoder, FairseqLanguageModel,
    register_model, register_model_architecture,
)

from .fconv import Embedding

from .pronouns import PronounLexicon

from math import log, exp


@register_model('refreader_gap_bert')
class RefReaderGapBertLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--emb-dropout', default=0.5, type=float, 
                            metavar='D',
                            help='dropout probability')
        parser.add_argument('--recurrent-dropout', default=0.5, type=float, 
                            metavar='D',
                            help='dropout probability')
        parser.add_argument('--output-dropout', default=0.5, type=float,
                            metavar='D',
                            help='dropout probability')
        parser.add_argument('--decoder-embed-dim', default=300, type=int,
                            metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-out-embed-dim', default=300, type=int, 
                            metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-layers', default=1, type=int, 
                            metavar='N',
                            help='number of layers in decoder LSTM')
        parser.add_argument('--bert-out-embed-dim', default=768, type=int,
                            metavar='N',
                            help='bert output embedding dimension')
        parser.add_argument('--bert-out-nb-layers', default=4, type=int,
                            metavar='N',
                            help='bert output number of layers')
        parser.add_argument('--bert-combine-mode', metavar='VAL', 
                            default='concat',
                            choices=['concat', 'sum', 'last'])
        parser.add_argument('--mem-cells', default=2, type=int, metavar='N',
                            help='size of fixed memory')
        parser.add_argument('--mem-key-size', default=32, type=int, metavar='N',
                            help='memory key embedding size')
        parser.add_argument('--gumbel-softmax', dest='gumbel_softmax', 
                            action='store_true')
        parser.add_argument('--no-gumbel-softmax', dest='gumbel_softmax', 
                            action='store_false')
        parser.set_defaults(gumbel_softmax=True)
        parser.add_argument('--gumbel-softmax-temperature', default=1.0, 
                            type=float, metavar='D', 
                            help='gumbel softmax temperature')
        parser.add_argument('--gumbel-softmax-temperature-anneal-interval', 
                            default=10, type=int, metavar='N',
                            help='gumbel softmax temperature anneal interval')
        parser.add_argument('--gumbel-softmax-temperature-anneal-rate', 
                            default=0.5, type=float, metavar='D', 
                            help='gumbel softmax temperature anneal rate')
        parser.add_argument('--halflife', default=30, type=int, 
                            metavar='N', help='halflife')
        parser.add_argument('--entity-halflife', default=4, type=int, 
                            metavar='N', help='entity halflife')
        parser.add_argument('--max-positions', default=1024, type=int, 
                            metavar='N', help='max positions')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        refreader_basic(args)
        decoder = RefReaderGapBertDecoder(
            token_dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            emb_dropout=args.emb_dropout,
            recurrent_dropout=args.recurrent_dropout,
            output_dropout=args.output_dropout,
            bert_out_embed_dim=args.bert_out_embed_dim,
            bert_out_nb_layers=args.bert_out_nb_layers,
            bert_combine_mode=args.bert_combine_mode,
            mem_cells=args.mem_cells,
            mem_key_size=args.mem_key_size,
            gumbel_softmax=args.gumbel_softmax,
            gumbel_softmax_temperature=args.gumbel_softmax_temperature,
            gumbel_softmax_temperature_anneal_interval=args.gumbel_softmax_temperature_anneal_interval,
            gumbel_softmax_temperature_anneal_rate=args.gumbel_softmax_temperature_anneal_rate,
            halflife=args.halflife,
            entity_halflife=args.entity_halflife,
            fp16=args.fp16,
        )
        return RefReaderGapBertLanguageModel(decoder)

class RefReaderGapBertDecoder(GapBertDecoder):
    def __init__(
            self, token_dictionary, 
            embed_dim=512, emb_dropout=0.5, recurrent_dropout=0.5, 
            output_dropout=0.5, bert_out_embed_dim=768, bert_out_nb_layers=4, 
            bert_combine_mode='concate', mem_cells=2, mem_key_size=32, 
            gumbel_softmax=True, 
            gumbel_softmax_temperature=1.0, 
            gumbel_softmax_temperature_anneal_interval=10,
            gumbel_softmax_temperature_anneal_rate=0.5, 
            halflife=30, entity_halflife=4,
            fp16=False):
        super().__init__(token_dictionary)

        self.emb_dropout = emb_dropout
        self.recurrent_dropout = recurrent_dropout
        self.output_dropout = output_dropout
        self.hidden_size = embed_dim
        self.bert_out_embed_dim = bert_out_embed_dim
        self.bert_out_nb_layers = bert_out_nb_layers
        self.bert_combine_mode = bert_combine_mode
        self.num_mem_cells = mem_cells
        self.mem_key_size = mem_key_size
        self.mem_val_size = embed_dim
        num_embeddings = len(token_dictionary)
        self.padding_idx = token_dictionary.pad()
        self.gumbel_softmax = gumbel_softmax
        self.gumbel_softmax_temperature = gumbel_softmax_temperature
        self.gumbel_softmax_temperature_anneal_interval = \
            gumbel_softmax_temperature_anneal_interval
        self.gumbel_softmax_temperature_anneal_rate = \
            gumbel_softmax_temperature_anneal_rate
        self._eps = 1e-7
        self.token_dictionary = token_dictionary

        self.fp16 = fp16

        if self.bert_combine_mode == "concat":
            gru_input_size = self.bert_out_nb_layers * self.bert_out_embed_dim
        elif self.bert_combine_mode in ["sum", "last"]:
            gru_input_size = self.bert_out_embed_dim

        self.layers = nn.ModuleList([
            nn.GRUCell(
                input_size=gru_input_size,
                hidden_size = embed_dim
            )
        ])

        self.init_hiddens = nn.Parameter(torch.FloatTensor(1, embed_dim))
        torch.nn.init.normal_(self.init_hiddens)
        self.init_salience = self._eps

        using_memory = (self.num_mem_cells > 0)
        self.init_mem_vals = nn.Parameter(
            torch.FloatTensor(self.num_mem_cells,self.mem_val_size),
            requires_grad = using_memory
        )
        self.init_mem_keys = nn.Parameter(
            torch.FloatTensor(self.num_mem_cells,self.mem_key_size),
            requires_grad = using_memory
        )
        torch.nn.init.normal_(self.init_mem_vals)
        torch.nn.init.normal_(self.init_mem_keys)

        self.mention_finder = nn.Sequential(
            nn.Linear(self.hidden_size,1), nn.LogSigmoid()
        )

        self.ref_mention_finder = nn.Sequential(
            nn.Linear(self.hidden_size,1), nn.LogSigmoid()
        )

        self.project_to_query = nn.Sequential(
            nn.Linear(self.hidden_size, self.mem_key_size),
            nn.Tanh(),
            nn.Linear(self.mem_key_size, self.mem_key_size)
        )
        
        # RNN output to key and value
        self.project_to_key_fc1 = nn.Linear(
            self.hidden_size, self.mem_key_size
        )
        self.project_to_key_fc2 = nn.Linear(
            self.mem_key_size, self.mem_key_size
        )
        
        self.project_to_val = nn.Sequential(
            nn.Linear(self.hidden_size,self.mem_val_size), nn.Tanh()
        )

        self.mem_key_and_val_updates = 'gru'

        self.key_update = nn.GRUCell(input_size = self.mem_key_size,
                                            hidden_size = self.mem_key_size)
        self.val_update = nn.GRUCell(input_size = self.mem_val_size,
                                            hidden_size = self.mem_val_size)

        # generic update function to be called on both key and value update
        self.do_update = lambda new_thing, old_things, updater : \
                            torch.stack([updater(new_thing, old_things[:,i,:])
                                        for i in range(self.num_mem_cells)],
                                        dim=2).transpose(1,2)
            
        self.query_match_offset = nn.Parameter(self.new_zeros(1))
        
        # hyperparameters of memory salience updates
        self.halflife = halflife 
        # salience decays by 50% after this number of tokens
        self.mem_persistence_no_ent = exp(log(.5) / self.halflife)
        # salience decays by 50% after 4 entities
        self.entity_halflife = entity_halflife 
        self.mem_persistence_ent = exp(log(.5) / self.entity_halflife)

        self.output_units = embed_dim
        self.out_projection = nn.Linear(self.output_units,len(token_dictionary))

        self.combine_gate = nn.Sequential(
            nn.Linear(self.hidden_size,1), nn.Sigmoid()
        )
        self.combine_pre_gru_in = nn.Linear(gru_input_size, self.hidden_size)
        self.combine_pre_gru_rec = nn.Linear(self.hidden_size, self.hidden_size)
        self.do_combine_pre_gru = \
            lambda x, y : torch.tanh(self.combine_pre_gru_in(x)
                            + self.combine_pre_gru_rec(y))


    def new_zeros(self,*args):
        return self.init_hiddens.data.new_zeros(args)
        
    def new_ones(self,*args):
        return self.init_hiddens.data.new_ones(args)
        
    def memory_update_gates(self, rnn_state, memory_keys, memory_vals, 
                            memory_salience):
        bsz = rnn_state.size(0)

        log_is_ent = self.mention_finder(rnn_state)
        # [bsz, 1]
        is_ent = torch.exp(log_is_ent)
        # [bsz, 1]

        is_referential = torch.exp(self.ref_mention_finder(rnn_state)) * is_ent
        # [bsz, 1]

        query = self.project_to_query(rnn_state)
        # [bsz, key_size]

        # compute compatibility of query and key, plus offset
        query_match = torch.bmm(
            memory_keys, query.view(bsz,-1,1)
        ).view(bsz,-1) + self.query_match_offset
        # [bsz, nb_mem_cells]

        query_match = F.softmax(query_match, dim=1) * is_referential
        # [bsz, nb_mem_cells]

        # for each memory, the update cannot exceed twice the prior salience
        update_gate = torch.min(query_match, 2 * memory_salience)
        # [bsz, nb_mem_cells]
        
        # what cannot be updated is written through an overwrite
        amount_to_write = is_ent - update_gate.sum(1).view(-1,1)
        # [bsz, 1]

        amount_to_write = torch.clamp(amount_to_write, min=0, max=1)
        # [bsz, 1]

        if self.gumbel_softmax and self.training:
            # only at training time, at test time, switch to argmin
            # memory_salience: [B, nb_memory_cells]
            temperature = self.gumbel_softmax_temperature
            if self.gumbel_softmax_temperature_anneal_interval != 0:
                # convert self._epoch to 0-based
                temperature *= self.gumbel_softmax_temperature_anneal_rate \
                                ** ((self._epoch - 1) // self.gumbel_softmax_temperature_anneal_interval)

            salience_memory_weights = F.gumbel_softmax(
                -memory_salience, tau=temperature
            )
            # [bsz, nb_mem_cells]

            overwrite_gate = amount_to_write.expand([-1, self.num_mem_cells]) * salience_memory_weights
            # [bsz, nb_mem_cells]

        else:
            # entire overwrite goes to the cell with the lowest salience
            min_salience_memories = memory_salience.argmin(1)

            overwrite_gate = self.new_zeros(bsz, self.num_mem_cells)
            # [bsz, nb_mem_cells]
            overwrite_gate[torch.arange(bsz), min_salience_memories] = amount_to_write.view(bsz)

        # each memory is kept to the extent that it's not updated or overwritten
        remember_gate = 1 - overwrite_gate - update_gate
        # [bsz, nb_mem_cells]

        remember_gate = torch.clamp(remember_gate, min=0, max=1)
        # [bsz, nb_mem_cells]

        return update_gate, overwrite_gate, remember_gate, query_match

    @staticmethod
    def compute_coref_mat_vectorized(overwrite, update, eps = 1e-7):
        """
        compute the coreference implied by the series of overwrite and update decisions

        Args:
        overwrite (FloatTensor, B x T x M): the overwrite gates
        update (FloatTensor, B x T x M): the update gates

        Returns:
        pred_coref (FloatTensor, B x T x T): the imputed coreference probabilities.

        each pred_coref[b,:,:] is an upper-right triangular matrix.

        see compute_coref_mat for a simpler non-vectorized version of this same function

        """
        bsz, seqlen, nb_mem_cells = overwrite.size()

        if overwrite.max() > 1:
            raise ValueError("numerical error in compute coref mat: overwrites cannot have probability 1")
        
        eps_clamp = lambda ten : torch.clamp(eps + ten, min=eps, max=1-eps)

        log_sum_retain = torch.cat([
            overwrite.new_zeros(bsz, 1, nb_mem_cells), 
            torch.cumsum(torch.log(eps_clamp(1 - overwrite)),1)
        ], 1)
        # [bsz, seqlen + 1, nb_mem_cells]

        # probability that the input at time t1 is stored in memory m
        # divide by prod_retain to account for dissipation of cumulative probability from 0:t1
        # i.e. p(retain t1 --> t2) = p(retain 0 --> t2) / p(retain 0 --> t1)
        # t1_stored = ((update+over)/prod_retain[:,1:,:])
        # the bmm operation sums across memory cells
        # pred_coref_vec = torch.bmm(t1_stored, (update * prod_retain[:,:-1,:]).transpose(1,2))

        # improved numerical stability
        t1_logsum = torch.log(eps_clamp(update + overwrite)) - log_sum_retain[:,1:,:]
        # [bsz, seqlen, nb_mem_cells]
        t2_logsum = torch.log(eps_clamp(update)) + log_sum_retain[:,:-1,:]
        # [bsz, seqlen, nb_mem_cells]

        pred_coref = overwrite.new_zeros(bsz, seqlen, seqlen)
        # [bsz, seqlen, seqlen]

        # vectorized, tested using np.array_equal below
        triu_mask = overwrite.new_ones(seqlen, seqlen)
        # [seqlen, seqlen]
        triu_mask = torch.triu(triu_mask, diagonal=1)
        # [seqlen, seqlen]
        triu_mask = triu_mask.view(1, seqlen, seqlen)
        # [seqlen, seqlen]

        pred_coref = torch.exp(torch.logsumexp(
            t1_logsum.unsqueeze(2) + t2_logsum.unsqueeze(1), 
            dim=3
        )) * triu_mask
        # [bsz, seqlen, seqlen]

        return pred_coref
                                        
    
    @staticmethod
    def compute_coref_mat(over,update):
        """
        compute the coreference implied by the series of overwrite and update decisions

        Args:
        over (FloatTensor, B x T x M): the overwrite gates
        update (FloatTensor, B x T x M): the update gates

        Returns:
        pred_coref (FloatTensor, B x T x T): the imputed coreference probabilities.

        each pred_coref[b,:,:] is an upper-right triangular matrix.

        see compute_coref_mat_vectorized for the fast version to use
        """

        #raise Error('obsolete, use compute_coref_mat_vectorized')
        B,T,M = over.size()
        pred_coref = over.data.new_zeros((B,T,T))

        # association of each state t with each memory m
        assoc = over + update
        for i in range(T):
            persist = over[:,i,:] + update[:,i,:]
            for j in range(i+1,T):
                # can pull this out of the loop?
                out_ij = torch.bmm(update[:,j,:].unsqueeze(1),
                                   persist.unsqueeze(2)).squeeze()
                pred_coref[:,i,j] = out_ij
                persist = persist * (1 - over[:,j,:])
        return pred_coref

    def forward(self, input_bert_emb):
        bert_emb = input_bert_emb
        # [bsz, nb_bert_layers, seqlen, bert_emb_size]
        bert_emb = bert_emb.permute(0, 2, 1, 3)
        # [bsz, seqlen, nb_bert_layers, bert_emb_size]
        bsz, seqlen, nb_bert_layers, bert_emb_size = bert_emb.size()
        bert_emb = torch.reshape(bert_emb, (bsz, seqlen, nb_bert_layers * bert_emb_size))
        # [bsz, seqlen, nb_bert_layers * bert_emb_size]

        # get word embeddings
        x = F.dropout(bert_emb, p=self.emb_dropout, training=self.training)
        # [bsz, seqlen, nb_bert_layers * bert_emb_size]

        x = x.transpose(0,1)
        # [seqlen, bsz, nb_bert_layers * bert_emb_size]

        # initialize hidden state of GRU
        prev_hidden = self.init_hiddens.repeat(bsz, 1)
        # [bsz, emb_size]

        # initialize memory
        memory_keys = self.init_mem_keys.repeat(bsz,1,1)
        # [bsz, nb_mem_cells, mem_key_size]
        memory_vals = self.init_mem_vals.repeat(bsz,1,1)
        # [bsz, nb_mem_cells, mem_val_size]

        # starting at zero leads to numerical instability
        memory_salience = self.init_salience * self.new_ones(bsz,self.num_mem_cells)
        # [bsz, nb_mem_cells]

        # list of GRU states, which will be pushed through output projection
        rnn_outs = []

        # keep track of gate operation for auxiliary loss
        overwrite_gates = []
        update_gates = []

        for j in range(seqlen):
            cur_x = x[j, :, :]
            # [bsz, nb_bert_layers * bert_emb_size]

            rnn = self.layers[0]
            # combine h_{t-1} and x_t
            # rnn_out: [B, emb_size]
            pre_recurrent_state = self.do_combine_pre_gru(cur_x, prev_hidden)
            # [bsz, emb_size]
        
            pre_recurrent_state = F.dropout(
                pre_recurrent_state, p=self.recurrent_dropout, 
                training=self.training
            )
            # [bsz, emb_size]
            
            # compute memory gates 
            update_gate, overwrite_gate, remember_gate, q_match \
                = self.memory_update_gates(pre_recurrent_state, memory_keys, memory_vals, memory_salience)

            is_ent = (update_gate + overwrite_gate).sum(1).view(-1,1)
            # [bsz, 1]

            # salience is update + overwrite + decayed previous ssalience
            memory_salience = update_gate + overwrite_gate\
                                + remember_gate * memory_salience\
                                * (is_ent * self.mem_persistence_ent\
                                + (1 - is_ent) * self.mem_persistence_no_ent)
            # [bsz, nb_mem_cells]

            # memory state is a weighted sum of the memory cells
            memory_state = torch.bmm(
                memory_salience.view(bsz,1,-1), memory_vals
            ).view(bsz,-1)
            # [bsz, emb_size]
            
            # gate the contribution of the memory to the GRU state
            ref_gate = self.combine_gate(pre_recurrent_state)
            # [bsz, 1]

            # this is the C-GRU update
            ref_gate = torch.min(
                ref_gate, # [B, 1]
                memory_salience.sum(dim=1, keepdim=True)
            )
            # [bsz, 1]

            rnn_state = (1. - ref_gate) * prev_hidden + ref_gate * memory_state
            # [bsz, emb_size]
            prev_hidden = rnn(cur_x, rnn_state)
            # [bsz, emb_size]
            rnn_out = F.dropout(
                prev_hidden, p=self.output_dropout, training=self.training
            )
            # [bsz, emb_size]

            # update the output and gate histories
            rnn_outs.append(rnn_out)
            overwrite_gates.append(overwrite_gate)
            update_gates.append(update_gate)

            ## now update the memory

            # compute overwrite candidates from little residual network
            residual = torch.tanh(self.project_to_key_fc1(
                    pre_recurrent_state
                )
            )
            # [bsz, key_size]
            key_overwrite_candidate = F.dropout(
                residual, p=self.recurrent_dropout, training=self.training
            )
            # [bsz, key_size]
                                    
            key_overwrite_candidate = torch.tanh(
                self.project_to_key_fc2(key_overwrite_candidate)
            )
            # [bsz, key_size]

            key_overwrite_candidate = residual + F.dropout(
                key_overwrite_candidate, p=self.recurrent_dropout,
                training=self.training
            )
            # [bsz, key_size]

            val_overwrite_candidate = self.project_to_val(pre_recurrent_state)
            # [bsz, val_size]
            
            # update candidates: combine the overwrite candidate and the previous key/value
            key_update_candidate = self.do_update(
                key_overwrite_candidate, memory_keys, self.key_update
            )
            # [bsz, nb_mem_cells, key_size]
            val_update_candidate = self.do_update(
                val_overwrite_candidate, memory_vals, self.val_update
            )
            # [bsz, nb_mem_cells, val_size]
            
            # new memory: weighted sum of prev value, overwrite, and update values
            memory_keys = remember_gate.view(bsz, -1, 1) * memory_keys\
                            + overwrite_gate.view(bsz, -1, 1) * key_overwrite_candidate.view(bsz,1,-1)\
                            + update_gate.view(bsz, -1, 1) * key_update_candidate
            # [bsz, nb_mem_cells, key_size]
        
            memory_vals = remember_gate.view(bsz, -1, 1) * memory_vals\
                            + overwrite_gate.view(bsz, -1, 1) * val_overwrite_candidate.view(bsz,1,-1)\
                            + update_gate.view(bsz, -1, 1) * val_update_candidate
            # [bsz, nb_mem_cells, val_size]

        rnn_outs = torch.cat(rnn_outs, dim=0).view(seqlen, bsz, -1).transpose(1,0)
        # [bsz, seqlen, emb_size]
        overwrite_gates = torch.cat(overwrite_gates, dim=0).view(seqlen, bsz,-1).transpose(1,0)
        # [bsz, seqlen, nb_memory_cells]
        update_gates = torch.cat(update_gates, dim=0).view(seqlen, bsz,-1).transpose(1,0)
        # [bsz, seqlen, nb_memory_cells]

        corefs = self.compute_coref_mat_vectorized(
            overwrite_gates, update_gates, self._eps
        )
        # [bsz, seqlen, seqlen]

        return self.out_projection(rnn_outs), corefs

    def max_positions(self):
        return int(400)

    # copied from lstm.py
    # it does seem to matter, although i can't tell what it does
    def hide_reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state[:2]))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

@register_model_architecture('refreader_gap_bert','refreader_gap_bert')
def refreader_basic(args):
    args.dropout=getattr(args,'dropout',0.5)
    args.decoder_embed_dim=getattr(args,'decoder_embed_dim',300)
    args.decoder_layers=getattr(args,'decoder_layers',1)
    args.mem_cells=getattr(args,'mem_cells',2)
    args.mem_key_size=getattr(args,'mem_key_size',32)
