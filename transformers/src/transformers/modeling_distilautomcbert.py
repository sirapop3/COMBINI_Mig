# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DistilBERT model
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
    and in part from HuggingFace PyTorch version of Google AI Bert model (https://github.com/google-research/bert)
"""


import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .activations import gelu
from .configuration_distilbert import DistilBertConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import PreTrainedModel, prune_linear_layer


logger = logging.getLogger(__name__)


DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "distilbert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-pytorch_model.bin",
    "distilbert-base-uncased-distilled-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-distilled-squad-pytorch_model.bin",
    "distilbert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-cased-pytorch_model.bin",
    "distilbert-base-cased-distilled-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-cased-distilled-squad-pytorch_model.bin",
    "distilbert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-german-cased-pytorch_model.bin",
    "distilbert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-multilingual-cased-pytorch_model.bin",
    "distilbert-base-uncased-finetuned-sst-2-english": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-finetuned-sst-2-english-pytorch_model.bin",
}


# UTILS AND BUILDING BLOCKS OF THE ARCHITECTURE #


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.

        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.output_attentions = config.output_attentions

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, query, key, value, mask, head_mask=None):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if self.output_attentions:
            return (context, weights)
        else:
            return (context,)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in ["relu", "gelu"], "activation ({}) must be in ['relu', 'gelu']".format(
            config.activation
        )
        self.activation = gelu if config.activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.output_attentions = config.output_attentions

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask, head_mask=head_mask)
        if self.output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if self.output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])


        self.linear_size = 32
        
        self.linear_1 = nn.ModuleList([nn.Linear(config.hidden_size, self.linear_size),
                                       nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size)
                                       ])

        self.linear_2 = nn.ModuleList([nn.Linear(self.linear_size, 1),
                                       nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1)
                                      ])

    def get_device_of(self, tensor):
        """
        Returns the device of the tensor.
        """
        if not tensor.is_cuda:
            return -1
        else:
            return tensor.get_device()

    def get_range_vector(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy COMBINI-data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
        else:
            return torch.arange(0, size, dtype=torch.long)

    def get_ones(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy COMBINI-data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.FloatTensor(size, device=device).fill_(1)
        else:
            return torch.ones(size)

    def get_zeros(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy COMBINI-data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).zero_()
        else:
            return torch.zeros(size, dtype=torch.long)

    def flatten_and_batch_shift_indices(self, indices, sequence_length):
        offsets = self.get_range_vector(indices.size(0), self.get_device_of(indices)) * sequence_length
        for _ in range(len(indices.size()) - 1):
            offsets = offsets.unsqueeze(1)

        # Shape: (batch_size, d_1, ..., d_n)
        offset_indices = indices + offsets

        # Shape: (batch_size * d_1 * ... * d_n)
        offset_indices = offset_indices.view(-1)
        return offset_indices

    def forward(self, x, attn_mask=None, head_mask=None, copy_rate=None, tokens_prob=None, question_ends_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = ()
        all_attentions = ()
        hidden_states = x 
        seq_mask = attn_mask.to(dtype=next(self.parameters()).dtype)

        ori_bsz, ori_num_items, _ = hidden_states.size()
        bsz = ori_bsz // 4

        device = self.get_device_of(hidden_states)
        w = 0
        tot_zoom = None

        tot_select_loss = 0
        Ls = []
        # a = time.time()
        # aa = 0
        mini = 1#16 
        mid = len(self.layer)//2
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            num_items = hidden_states.size(1)

            # if (i in [1, 4, 7]) and (num_items > mini):
            if (i in [1, mid]) and (num_items > mini):
                h1 = gelu(self.linear_1[w](hidden_states))
                h2 = self.linear_2[w](h1).squeeze(-1)
                h2 = h2.view(bsz, -1, num_items)
                h2 = torch.mean(h2, dim=1)
                prob = torch.sigmoid(h2) * question_ends_mask
                max_seq_mask = (torch.sum(seq_mask.view(bsz, -1, num_items), dim=1) > 0).to(seq_mask)

                max_seq_L = torch.sum(question_ends_mask, dim=-1)
                option_l = torch.sum(max_seq_mask - question_ends_mask, dim=-1)

                if copy_rate is not None:
                    if copy_rate.sum()>0:

                        prob_plus = prob

                        with torch.no_grad():
                            if tot_zoom is None:
                                guide = tokens_prob[:,w,:]   # (bsz, 8, seq_len)
                            else:
                                guide = torch.matmul(tot_zoom, tokens_prob[:, w, :].unsqueeze(-1)).squeeze(-1)

                            guide = guide * question_ends_mask
                            prob_L = torch.sum(prob_plus, dim=-1)

                            prob_L = (prob_L+0.5).int()
                            num_items_to_keep = torch.max(prob_L)

                            top_values, top_indices = guide.topk(num_items_to_keep, 1)                    
                            flatten_indices = self.flatten_and_batch_shift_indices(top_indices, num_items)

                            if device > -1:
                                rt = torch.cuda.FloatTensor(bsz, num_items_to_keep, device=device).zero_()
                                t_p = torch.cuda.FloatTensor(bsz * num_items, device=device).zero_()
                            else:
                                rt = torch.zeros(bsz, num_items_to_keep)
                                t_p = torch.zeros(bsz * num_items)

                            for k in range(bsz):
                                rt[k, :prob_L[k]] = 1

                            t_p[flatten_indices] = rt.view(-1)#1
                            t_p = t_p.view(bsz, num_items)

                        copy_mask = copy_rate.unsqueeze(-1)
                        caiyang_prob = prob_plus * (1-copy_mask) + t_p * copy_mask
                        
                    else:
                        caiyang_prob = prob


                    caiyang_prob = caiyang_prob.clone()
                    caiyang_prob[:, 0] = 1

                    m = torch.distributions.bernoulli.Bernoulli(caiyang_prob)               
                    selected_token = m.sample() * max_seq_mask



                    selected_token =  ((selected_token + max_seq_mask - question_ends_mask)>0).to(selected_token)

                    select_loss = - ( selected_token * torch.log(prob+1e-6) + (1 - selected_token) * torch.log(1 - prob +1e-6 ) ) 
                    select_loss = select_loss * question_ends_mask
                    select_loss = torch.sum(select_loss, dim=-1) /  torch.sum(question_ends_mask, dim=-1)         
                    # assert(train_layer is not None)
                    # if train_layer is None or train_layer==w:
                    tot_select_loss += select_loss

                else:
                    caiyang_prob = prob.clone()
                    caiyang_prob += max_seq_mask - question_ends_mask
                    caiyang_prob[:, 0] = 3
                    selected_token = (caiyang_prob >= 0.5).to(max_seq_mask) * max_seq_mask


                l = torch.sum(selected_token, dim=-1) 
                Ls.append(l / ori_num_items)

                # num_items_to_keep = int((torch.max(l+option_l)).item())

                num_items_to_keep = int((torch.max(l)).item())

                with torch.no_grad():
                    if copy_rate is None: 
                        top_values, top_indices = caiyang_prob.topk(num_items_to_keep, 1)    
                    else:
                        selected_token_2 = selected_token + max_seq_mask - question_ends_mask
                        selected_token_2[:, 0] = 3

                        top_values, top_indices = selected_token_2.topk(num_items_to_keep, 1)    

                    if device > -1:
                        zoomMatrix = torch.cuda.FloatTensor(bsz * num_items_to_keep, num_items, device=device).zero_()
                    else:
                        zoomMatrix = torch.zeros(bsz * num_items_to_keep, num_items)

                    idx = self.get_range_vector(bsz*num_items_to_keep, device)
                    zoomMatrix[idx, top_indices.view(-1)] = 1.
                    zoomMatrix = zoomMatrix.view(bsz, num_items_to_keep, num_items)

                    if copy_rate is not None and copy_rate[0]>0:
                        if tot_zoom is None:
                            tot_zoom = zoomMatrix
                        else:
                            tot_zoom = torch.matmul(zoomMatrix, tot_zoom)

                zoomMatrix_expand = zoomMatrix.view(1, bsz, num_items_to_keep, num_items).expand(4, -1, -1, -1)
                zoomMatrix_expand = zoomMatrix_expand.transpose(0, 1)
                zoomMatrix_expand = zoomMatrix_expand.reshape(ori_bsz, num_items_to_keep, num_items)


                seq_mask = torch.matmul(zoomMatrix_expand, seq_mask.unsqueeze(-1))
                seq_mask = seq_mask.squeeze(-1) 

                question_ends_mask = torch.matmul(zoomMatrix, question_ends_mask.unsqueeze(-1))
                question_ends_mask = question_ends_mask.squeeze(-1) 

                if copy_rate is not None:
                    top_values = (top_values>0).to(seq_mask)
                    question_ends_mask = question_ends_mask * top_values
                    top_values_expand = top_values.view(1, bsz, num_items_to_keep).expand(4, -1, -1)
                    top_values_expand = top_values_expand.transpose(0, 1)
                    top_values_expand = top_values_expand.reshape(ori_bsz, num_items_to_keep)

                    seq_mask = seq_mask * top_values_expand

                hidden_states = torch.matmul(zoomMatrix_expand, hidden_states)       
                w += 1

            attn_mask = seq_mask.long()
            layer_outputs = layer_module(x=hidden_states, attn_mask=attn_mask, head_mask=head_mask[i])

            hidden_states = layer_outputs[0]
            

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)            

            # if self.output_attentions:
            #     all_attentions = all_attentions + (layer_outputs[1],)

        outputs = (hidden_states, tot_select_loss, Ls)

        # outputs = (hidden_state,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


# INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL #
class DistilBertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DistilBertConfig
    pretrained_model_archive_map = DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = None
    base_model_prefix = "distilbert"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


DISTILBERT_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.DistilBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

DISTILBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.DistilBertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top.",
    DISTILBERT_START_DOCSTRING,
)
class AUTOMCDistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = Embeddings(config)  # Embeddings
        self.transformer = Transformer(config)  # Encoder

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, copy_rate=None, tokens_prob=None, question_ends_mask=None):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.DistilBertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import DistilBertTokenizer, DistilBertModel
        import torch

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        model = DistilBertModel.from_pretrained('distilbert-base-cased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
        tfmr_output = self.transformer(x=inputs_embeds, attn_mask=attention_mask, head_mask=head_mask,
                    copy_rate=copy_rate, tokens_prob=tokens_prob, question_ends_mask=question_ends_mask)
        hidden_state = tfmr_output[0]
        output = (hidden_state,) + tfmr_output[1:]

        return output  # last-layer hidden-state, (all hidden_states), (all attentions)


@add_start_docstrings(
    """DistilBert Model with a `masked language modeling` head on top. """, DISTILBERT_START_DOCSTRING,
)
class DistilBertForMaskedLM(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.init_weights()

        self.mlm_loss_fct = nn.CrossEntropyLoss()

    def get_output_embeddings(self):
        return self.vocab_projector

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, masked_lm_labels=None):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.DistilBertConfig`) and inputs:
        loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import DistilBertTokenizer, DistilBertForMaskedLM
        import torch

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        model = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

        """
        dlbrt_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        outputs = (prediction_logits,) + dlbrt_output[1:]
        if masked_lm_labels is not None:
            mlm_loss = self.mlm_loss_fct(
                prediction_logits.view(-1, prediction_logits.size(-1)), masked_lm_labels.view(-1)
            )
            outputs = (mlm_loss,) + outputs

        return outputs  # (mlm_loss), prediction_logits, (all hidden_states), (all attentions)



@add_start_docstrings(
    """DistilBert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    DISTILBERT_START_DOCSTRING,
)
class DistilBertForTokenClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING)
    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.DistilBertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import DistilBertTokenizer, DistilBertForTokenClassification
        import torch

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

        """

        outputs = self.distilbert(
            input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)



class AUTOMCDistilBertForMultipleChoice(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = AUTOMCDistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, 1)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.temperature = 1
        self.num_labels = 1
        self.init_weights()

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING.format("(batch_size, num_choices, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        copy_rate=None,
        tokens_prob=None,
        question_ends=None,
        teacher_logits=None

        # output_attentions=None,
        # output_hidden_states=None,
    ):
    
        # num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        bsz, num_choices, seq_len =  input_ids.shape

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        question_ends_mask = torch.zeros(bsz, seq_len).to(dtype=next(self.parameters()).dtype, device=input_ids.device)
        for i in range(bsz):
            question_ends_mask[i, :question_ends[i]] = 1.

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            copy_rate=copy_rate,
            tokens_prob=tokens_prob,
            question_ends_mask=question_ends_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
        )
        if copy_rate is not None:
            selector_loss = outputs[1]

        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        pooled_output = self.dropout(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)  # (bs * num_choices, 1)

        reshaped_logits = logits.view(-1, num_choices)  # (bs, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if teacher_logits is not None:
            teacher_probs = nn.functional.softmax(teacher_logits / self.temperature)
            student_log_probs = nn.functional.log_softmax(reshaped_logits / self.temperature)
            loss = -student_log_probs * teacher_probs

            loss = torch.sum(loss, dim=-1)

            # loss_fct = CrossEntropyLoss(reduction='none')
            # loss_ori = loss_fct(reshaped_logits, labels)

            outputs = (loss, selector_loss, loss) + outputs

        elif labels is not None:            
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss, selector_loss) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)