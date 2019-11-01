import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor

from fairseq import options, utils

from . import (
    FairseqIncrementalDecoder, FairseqLanguageModel,
    GapBertDecoder,
    register_model, register_model_architecture,
)

from .fconv import Embedding
from .lstm import LSTM, Linear, LSTMCell


@register_model('lstm_cache_lm')
class LSTMCacheLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--decoder-embed-dim',type=int,metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-out-embed-dim',type=int,metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-layers',type=int,metavar='N',
                            help='number of layers in decoder LSTM')
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        lstm_lm_basic(args)
        decoder = LSTMCacheDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout=args.dropout
        )
        return LSTMCacheLanguageModel(decoder)


class LSTMCacheDecoder(GapBertDecoder):
    """
    An LSTM decoder with a cache, using nn.LSTM and no attention
    """
    def __init__(
            self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
            dropout=0.1, start_at_zeros=False, pretrained_embed=None, use_cache=True):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.layers = nn.ModuleList([
            LSTMCell(
                input_size = embed_dim if layer == 0 else hidden_size,
                hidden_size = hidden_size
            )
            for layer in range(num_layers)
        ])

        self.output_units = hidden_size
        self.out_projection = nn.Linear(self.output_units,len(dictionary))

        self.init_hiddens = nn.Parameter(torch.FloatTensor(num_layers,hidden_size))
        self.init_cells = nn.Parameter(torch.FloatTensor(num_layers,hidden_size))

        torch.nn.init.normal_(self.init_hiddens)
        torch.nn.init.normal_(self.init_cells)

        if use_cache:
            #self.cache_gate = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),
            #                                nn.ReLU(),
            #                                nn.Linear(self.hidden_size,1),
            #                                nn.Sigmoid())
            self.alpha = 0.
            self.theta = 0.5
        
    def new_zeros(self,*args):
        return self.init_hiddens.data.new_zeros(args)
        
    def new_ones(self,*args):
        return self.init_hiddens.data.new_ones(args)

    def reset_init_params(self):
        for i in range(self.num_layers):
            nn.init.normal_(self.init_hiddens[i])
            nn.init.normal_(self.init_cells[i])

    def forward(self, prev_output_tokens, incremental_state = None):
        bsz,seqlen = prev_output_tokens.size()
        
        x = self._embed_tokens(prev_output_tokens, incremental_state)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0,1)

        tok_tensor = self.new_zeros(bsz,seqlen,len(self.dictionary.indices))
        indexer = torch.range(0,bsz-1).long()
        
        prev_hiddens = [layer[0] for layer in self.init_hiddens.repeat(bsz,1,1).transpose(0,1).split(1)]
        prev_cells = [layer[0] for layer in self.init_cells.repeat(bsz,1,1).transpose(0,1).split(1)]

        # hold init_hidden to start
        h_states = [self.init_hiddens.repeat(bsz,1,1).transpose(0,1)]
        for j,x_j in enumerate(x):
            tok_tensor[indexer,j,prev_output_tokens[:,j]] = 1.
            h_in = x_j
            for i, rnn in enumerate(self.layers):
                prev_hiddens[i], prev_cells[i] = rnn(h_in, (prev_hiddens[i], prev_cells[i]))
                h_in = F.dropout(prev_hiddens[i], p=self.dropout, training=self.training)
            h_states.append(h_in.view(1,bsz,-1))

        # back to B x T x C
        h_states = torch.cat(h_states[:-1],0).transpose(1,0)            
        rnn_lm = self.out_projection(h_states)

        # result: B x T x T
        pre_attn = torch.exp(self.theta * torch.bmm(h_states,h_states.transpose(2,1)) + self.alpha)
        attn = pre_attn.data.new_zeros(bsz,seqlen,seqlen)
        for b in range(bsz):
            attn[b,:,:] = torch.triu(pre_attn[b,:,:],diagonal=1)
        # rows will be source
        # cols will be target
        cache_lm = torch.bmm(attn.transpose(2,1),tok_tensor)

        print(attn.max(), cache_lm.max(), rnn_lm.max())

        if not self.training:
            return cache_lm + rnn_lm, None
        
        return rnn_lm, None

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    # not sure why we need this, but we do
    def max_positions(self):
        return int(1e8)

    def gate_loss(self):
        return self.new_zeros(1)
        
@register_model_architecture('lstm_cache_lm','lstm_cache_lm')
def lstm_lm_basic(args):
    args.dropout=getattr(args,'dropout',0.1)
    args.decoder_embed_dim=getattr(args,'decoder_embed_dim',1024)
    args.decoder_out_embed_dim=getattr(args,'decoder_out_embed_dim',1024)
    args.decoder_layers=getattr(args,'decoder_layers',1)
    return args

