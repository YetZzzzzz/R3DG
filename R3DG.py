import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch.nn import L1Loss, MSELoss
from torch.autograd import Function
from math import pi, log
from functools import wraps
from torch import nn, einsum
import torch.nn.functional as F
import os
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.activations import gelu, gelu_new
from transformers import BertConfig
import numpy as np
import torch.optim as optim
from itertools import chain
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from modules.transformer import TransformerEncoder
from torch.nn import CosineSimilarity


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device("cuda:0")

# MOSI SETTING
# ACOUSTIC_DIM = 74
# VISUAL_DIM = 35 # MOSI 47, MOSEI 35
# TEXT_DIM = 768
# MOSI SETTING
# ACOUSTIC_DIM = 74
# VISUAL_DIM = 35
# TEXT_DIM = 768
logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased"
]


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
BertLayerNorm = torch.nn.LayerNorm

ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "gelu_new": gelu_new,
    "mish": mish,
}
    
class R3DG_BertModel(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.config.output_hidden_states=True
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.d_l = args.d_l
        self.linear = nn.Linear(in_features=args.TEXT_DIM, out_features=self.d_l)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # fused_embedding = embedding_output

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_sequence_output = encoder_outputs[0]# 36*60*768:bsz*msl*dim
        text_outputs = self.linear(last_sequence_output[:,0])
    
        return text_outputs

class Audio_Video_network(nn.Module): 
    # with transformer encoder
    def __init__(self, modality_dim, grans, args = None):
        super(Audio_Video_network, self).__init__()
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.d_l = args.d_l
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_dropout = args.attn_dropout
        self.modality_dim = modality_dim # for 
        self.grans0 = grans[0]
        self.grans1 = grans[1]
        self.grans2 = grans[2]
        self.grans3 = grans[3]
        self.projs = nn.Conv1d(self.modality_dim, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)#
        self.avgmaxpoolings_0 = nn.AdaptiveMaxPool1d(self.grans0)
        self.avgmaxpoolings_1 = nn.AdaptiveMaxPool1d(self.grans1)
        self.avgmaxpoolings_2 = nn.AdaptiveMaxPool1d(self.grans2)
        self.avgmaxpoolings_3 = nn.AdaptiveMaxPool1d(self.grans3)
        self.avgmaxpoolings_all = nn.AdaptiveMaxPool1d(1)# if the original is 
        
        self.encoder = TransformerEncoder(embed_dim=self.d_l,
                                  num_heads= self.num_heads,
                                  layers=self.layers,
                                  attn_dropout= self.attn_dropout,
                                  relu_dropout=self.relu_dropout,   
                                  res_dropout= self.res_dropout,    
                                  embed_dropout=self.embed_dropout,  
                                  attn_mask= False)                 
    def forward(self,feas):
        # the modality dimension can not be divided by num_heads, so first use Conv1d to change the dimensions
        feas = feas.transpose(1, 2)
        feas = self.projs(feas) # , self.modality_dim
        feas = feas.permute(2, 0, 1)
        feas = self.encoder(feas)# output: [src_len, batch, modality] # [4, 60, 36, 64]
        feas = feas.permute(1,2,0)
        feas0 = self.avgmaxpoolings_0(feas)
        feas1 = self.avgmaxpoolings_1(feas)
        feas2 = self.avgmaxpoolings_2(feas)
        feas3 = self.avgmaxpoolings_3(feas)
        feas_all = self.avgmaxpoolings_all(feas)# 
        feas_all = feas_all.view(-1,self.d_l)
        feas_local = torch.cat((feas0, feas1,feas2, feas3), dim=2)
        
        return feas_local, feas_all
    


class Topk_search_rank(nn.Module):
    def __init__(self, selected_locals, args=None):# layers,0.1
        super(Topk_search_rank, self).__init__()
        self.dims = args.d_l
        self.sele_locals = selected_locals
        self.softmax = nn.Softmax(dim=1)

    def forward(self, modal1, modal2):
        # modal1 [bsz,d_l], modal2 [bsz, locals, d_l]
        # compute the similarity
        weights = nn.Parameter(torch.randn(modal2.shape[0], self.sele_locals,device=modal2.device).requires_grad_())
        modal1 = F.normalize(modal1, dim=-1)
        modal2 = F.normalize(modal2, dim=-1)
        modal1_expand = modal1.unsqueeze(1).expand(modal2.shape[0], modal2.shape[1], self.dims) # 
        cos_sim = F.cosine_similarity(modal1_expand, modal2, dim=2)
        # choose the top k
        topk_values, topk_indices = torch.topk(cos_sim, self.sele_locals, dim=1)
        # select the top k local representations
        selected_vectors = torch.gather(modal2, 1, topk_indices.unsqueeze(2).expand(modal2.shape[0], self.sele_locals, self.dims))# to see is modal2.shape[1] or self.sele_locals
        # initiate the weights
        weights_softmaxed = self.softmax(weights)
        # fuse the representations
        weighted_sum = torch.sum(selected_vectors * weights_softmaxed.unsqueeze(2), dim=1)
        return weighted_sum


# Attention implementation     
class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):# layers,0.1
        super(Attention, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.attention_mlp = nn.Sequential()
        self.attention_mlp.add_module('attention_mlp', nn.Linear(in_features=dim*2, out_features=hidden_dim))
        self.attention_mlp.add_module('attention_mlp_dropout', nn.Dropout(dropout))
        self.attention_mlp.add_module('attention_mlp_activation', nn.ReLU())
        self.fc_att = nn.Linear(hidden_dim, 2)

    def forward(self, feas1, feas2):
        multi_hidden1 = torch.cat([feas1, feas2], dim=1) # [bsz, 768*2]
        attention = self.attention_mlp(multi_hidden1) # [bsz, 64]  
        attention = self.fc_att(attention)# [bsz, 2]
        attention = torch.unsqueeze(attention, 2) * self.scale # [bsz, 2, 1]
        attention = attention.softmax(dim = 1)
        multi_hidden2 = torch.stack([feas1, feas2], dim=2) # [bsz, 768, 2]
        fused_feat = torch.matmul(multi_hidden2, attention) # 
        fused_feat = fused_feat.squeeze() # [bsz, 64]
        fused_feat = fused_feat.view(-1,self.dim)
        return fused_feat


class R3DG(BertPreTrainedModel):
    def __init__(self, config, args = None):
        super().__init__(config)
        self.num_labels = config.num_labels# here is 1
        self.d_l = args.d_l
        self.bert = R3DG_BertModel(config, args) #.d_l
        self.dropout = args.dropout_prob#nn.Dropout()
        self.activation = nn.ReLU()
        self.num_heads = args.num_heads
        self.audio_network = Audio_Video_network(args.ACOUSTIC_DIM, args.grans_a, args)# modality_dim, modality_length, grans,
        self.video_network = Audio_Video_network(args.VISUAL_DIM, args.grans_v, args)# modality_dim, modality_length, grans,
        self.search_a = Topk_search_rank(args.local_as, args=args)
        self.search_v = Topk_search_rank(args.local_vs, args=args)
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v', nn.Linear(in_features=self.d_l, out_features=self.d_l))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a', nn.Linear(in_features=self.d_l, out_features=self.d_l))
        self.alpha = args.alpha
        self.loss_recon = nn.MSELoss()
        
        encoder_layer_va = nn.TransformerEncoderLayer(d_model=self.d_l, nhead=self.num_heads)
        self.transformer_encoder_va = nn.TransformerEncoder(encoder_layer_va, num_layers=2) # num_layers
        
        self.fusion_va = nn.Sequential()
        self.fusion_va.add_module('fusion_layer_va', nn.Linear(in_features=self.d_l*2, out_features=self.d_l))
        self.fusion_va.add_module('fusion_layer_va_dropout', nn.Dropout(self.dropout))
        self.fusion_va.add_module('fusion_layer_va_activation', self.activation)
        
        encoder_layer_all = nn.TransformerEncoderLayer(d_model=self.d_l, nhead=self.num_heads)
        self.transformer_encoder_all = nn.TransformerEncoder(encoder_layer_all, num_layers=2) # num_layers
        
        self.fusion_all = nn.Sequential()
        self.fusion_all.add_module('fusion_layer_all', nn.Linear(in_features=self.d_l*2, out_features=self.d_l))
        self.fusion_all.add_module('fusion_layer_all_dropout', nn.Dropout(self.dropout))
        self.fusion_all.add_module('fusion_layer_all_activation', self.activation)
        self.fusion_all.add_module('fusion_layer_all_all', nn.Linear(in_features=self.d_l, out_features= self.num_labels)) 
        
        self.init_weights()

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        label_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        global_texts = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        local_audios, global_audios = self.audio_network(acoustic)
        local_videos, global_videos = self.video_network(visual)
        global_audios = global_audios.squeeze()
        global_videos = global_videos.squeeze()
        local_audios = local_audios.transpose(1, 2)
        local_videos = local_videos.transpose(1, 2)
        
        ## search and fuse the local representations
        sum_local_audios = self.search_a(global_texts,local_audios)
        sum_local_videos = self.search_v(global_texts,local_videos)
        ## compute the reconstruction loss
        local_audios_recon = self.recon_a(sum_local_audios)
        local_videos_recon = self.recon_v(sum_local_videos)
        loss_recon_a = self.loss_recon(local_audios_recon, global_audios)
        loss_recon_v = self.loss_recon(local_videos_recon, global_videos)


        sum_a_v = torch.stack((sum_local_audios, sum_local_videos), dim=0)
        sum_a_vs = self.transformer_encoder_va(sum_a_v)
        h_a_vs = torch.cat((sum_a_vs[0], sum_a_vs[1]), dim=1)
        h_a_vs = self.fusion_va(h_a_vs)

        sum_all = torch.stack((global_texts, h_a_vs), dim=0)
        sum_alls = self.transformer_encoder_all(sum_all)
        sum_alls = torch.cat((sum_alls[0], sum_alls[1]), dim=1)
        logits = self.fusion_all(sum_alls)

        all_losses = (loss_recon_a + loss_recon_v) * self.alpha
        
        return all_losses, logits, sum_alls


    def test(self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,):
        
        global_texts = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        local_audios, global_audios = self.audio_network(acoustic)
        local_videos, global_videos = self.video_network(visual)
        global_audios = global_audios.squeeze()
        global_videos = global_videos.squeeze()
        local_audios = local_audios.transpose(1, 2)
        local_videos = local_videos.transpose(1, 2)
        ## search and fuse the local representations
        sum_local_audios = self.search_a(global_texts,local_audios)
        sum_local_videos = self.search_v(global_texts,local_videos)

        sum_a_v = torch.stack((sum_local_audios, sum_local_videos), dim=0)
        sum_a_vs = self.transformer_encoder_va(sum_a_v)
        h_a_vs = torch.cat((sum_a_vs[0], sum_a_vs[1]), dim=1)
        h_a_vs = self.fusion_va(h_a_vs)

        sum_all = torch.stack((global_texts, h_a_vs), dim=0)
        sum_alls = self.transformer_encoder_all(sum_all)
        sum_alls = torch.cat((sum_alls[0], sum_alls[1]), dim=1)
        logits = self.fusion_all(sum_alls)
        
        return logits, sum_alls

  
