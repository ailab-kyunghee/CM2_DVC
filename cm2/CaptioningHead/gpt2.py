# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch OpenAI GPT-2 model."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import load_tf_weights_in_gpt2, GPT2LMHeadModel, GPT2MLP, GPT2Attention, GPT2Block, GPT2Model 

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
# from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    logging,
)
# from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
# from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import GPT2Config


if version.parse(torch.__version__) >= version.parse("1.6"):
    is_amp_available = True
    from torch.cuda.amp import autocast
else:
    is_amp_available = False

    
class ThisGPT2Config(GPT2Config):
    model_type = "this_gpt2"

    def __init__(
        self,
        cross_attention_reduce_factor = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_attention_reduce_factor = cross_attention_reduce_factor
        
class ThisGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

        #print("this gpt2")

        #print("self.is_cross_attention = is_cross_attention", self.is_cross_attention, is_cross_attention)
        
        self.cross_attention_reduce_factor = config.cross_attention_reduce_factor
        
        if self.is_cross_attention:
            self.c_attn = Conv1D(int(2 / self.cross_attention_reduce_factor * self.embed_dim), 
                                                                                  self.embed_dim) 
            self.q_attn = Conv1D(int(self.embed_dim / self.cross_attention_reduce_factor), self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim, int(self.embed_dim / self.cross_attention_reduce_factor))
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            #print("encoder_hidden provided######################################################################")
            #print("encoder_hidden_states: ",encoder_hidden_states.shape)
            
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            split_size = int(self.split_size / self.cross_attention_reduce_factor)
            head_dim = int(self.head_dim / self.cross_attention_reduce_factor)

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(split_size, dim=2)
            attention_mask = encoder_attention_mask
            #print("key: ",key.shape)
            #print("value: ",value.shape)
            query = self._split_heads(query, self.num_heads, head_dim)
            key = self._split_heads(key, self.num_heads, head_dim)
            value = self._split_heads(value, self.num_heads, head_dim)
            
        else:
            #print("######################################################################encoder_hidden not provided")
            
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        attn_output = self._merge_heads(attn_output, self.num_heads, int(self.head_dim / self.cross_attention_reduce_factor))
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class ThisGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        hidden_size = config.hidden_size
        print("########################################################################################",hidden_size)
        if config.add_cross_attention:
            self.crossattention = ThisGPT2Attention(config, is_cross_attention=False, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

class ThisGPT2Model(GPT2Model):

    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([ThisGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])


class ThisGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = ThisGPT2Config
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = ThisGPT2Model(config)











import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

class GPT2Captioner(nn.Module):
    def __init__(self, opt):
        super(GPT2Captioner, self).__init__()
        self.opt = opt

        self.vocab_size = opt.vocab_size
        self.max_caption_len = opt.max_caption_len

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2Model.from_pretrained('gpt2')

        self.logit = nn.Linear(self.model.config.hidden_size, self.vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.logit(hidden_states)
        return logits

    def generate(self, input_ids, attention_mask):
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=self.max_caption_len)
        generated_ids = outputs[:, input_ids.shape[1]:]
        return generated_ids

    def build_loss(self, logits, target_ids):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.vocab_size), target_ids.view(-1))
        return loss

    def tokenize(self, captions):
        input_ids = []
        attention_mask = []
        for caption in captions:
            encoded = self.tokenizer.encode_plus(caption, add_special_tokens=True, padding='max_length', max_length=self.max_caption_len, truncation=True)
            input_ids.append(encoded['input_ids'])
            attention_mask.append(encoded['attention_mask'])
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        return input_ids, attention_mask

    def detokenize(self, generated_ids):
        generated_captions = []
        for ids in generated_ids:
            caption = self.tokenizer.decode(ids, skip_special_tokens=True)
            generated_captions.append(caption)
        return generated_captions




import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

class Captioner(nn.Module):
    def __init__(self, opt):
        super(Captioner, self).__init__()
        self.opt = opt

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob
        self.max_caption_len = opt.max_caption_len

        self.ss_prob = 0.0 # Schedule sampling probability
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2Model.from_pretrained('gpt2')

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def build_loss(self, input, target, mask):
        one_hot = torch.nn.functional.one_hot(target, self.opt.vocab_size+1)
        max_len = input.shape[1]
        output = - (one_hot[:, :max_len] * input * mask[:, :max_len, None]).sum(2).sum(1) / (mask.sum(1) + 1e-6)
        return output

    def forward(self, hs, reference, others, cap_tensor):
        seq = cap_tensor
        vid_num, query_num, _ = hs.shape
        assert vid_num == 1

        reference_points = reference
        input_flatten = others['memory']
        input_spatial_shapes = others['spatial_shapes']
        input_level_start_index = others['level_start_index']
        input_padding_mask = others['mask_flatten']
        # Rest of the code...

    def get_logprobs_state(self, it, state, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, mask):
        xt = self.tokenizer.encode(it)
        xt = torch.tensor(xt).unsqueeze(0)
        output = self.model(xt)
        logprobs = F.log_softmax(self.logit(self.dropout(output)), dim=1)
        return logprobs, state

    def sample(self, hs, reference, others, opt={}):
        # Rest of the code...












import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

class Captioner(nn.Module):
    def __init__(self, opt):
        super(Captioner, self).__init__()
        self.opt = opt

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob
        self.max_caption_len = opt.max_caption_len

        self.ss_prob = 0.0 # Schedule sampling probability
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def build_loss(self, input, target, mask):
        one_hot = torch.nn.functional.one_hot(target, self.opt.vocab_size+1)
        max_len = input.shape[1]
        output = - (one_hot[:, :max_len] * input * mask[:, :max_len, None]).sum(2).sum(1) / (mask.sum(1) + 1e-6)
        return output

    def forward(self, hs, reference, others, cap_tensor):
        seq = cap_tensor
        vid_num, query_num, _ = hs.shape
        assert vid_num == 1

        # Use GPT-2 to encode the input sequence
        input_ids = seq.squeeze(0).tolist()
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_embeds = self.gpt2.transformer.wte(input_ids)
        input_embeds = input_embeds.transpose(0, 1)

        # Pass the encoded input through linear layer and softmax
        output = self.logit(self.dropout(input_embeds))
        output = torch.log_softmax(output, dim=-1)

        return output

    def sample(self, hs, reference, others, opt={}):
        vid_num, query_num, _ = hs.shape
        assert vid_num == 1
        batch_size = vid_num * query_num
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        # Use GPT-2 to generate captions
        input_ids = torch.tensor([[self.vocab_size]]).repeat(batch_size, 1)
        input_embeds = self.gpt2.transformer.wte(input_ids)
        input_embeds = input_embeds.transpose(0, 1)

        # Pass the generated input through linear layer and softmax
        output = self.logit(self.dropout(input_embeds))
        output = torch.log_softmax(output, dim=-1)

        # Sample words from the output
        if sample_max:
            sampleLogprobs, it = torch.max(output.data, 2)
            it = it.view(-1).long()
        else:
            if temperature == 1.0:
                prob_prev = torch.exp(output.data)
            else:
                prob_prev = torch.exp(torch.div(output.data, temperature))
            it = torch.multinomial(prob_prev, 1)
            sampleLogprobs = output.gather(2, it)
            it = it.view(-1).long()

        # Return the generated captions and their log probabilities
        seq = it.view(batch_size, -1)
        seqLogprobs = sampleLogprobs.view(batch_size, -1)

        return seq, seqLogprobs
