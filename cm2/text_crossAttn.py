import torch
import torch.nn as nn
from typing import Tuple
import sys


class CustomCrossAttention(nn.Module):
    def __init__(self, hidden_size,num_attention_heads):
        super(CustomCrossAttention, self).__init()

        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_attn = nn.Conv1d(2 * self.head_dim, self.embed_dim, 1)
        self.q_attn = nn.Conv1d(self.head_dim, self.embed_dim, 1)
        self.c_proj = nn.Conv1d(self.embed_dim, self.head_dim, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.FloatTensor,
    ) -> Tuple[torch.Tensor, ...]:

        query = self.q_attn(hidden_states)
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, None)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, head_dim):
        tensor = tensor.view(tensor.size(0), -1, num_heads, head_dim)
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(-1, tensor.size(-2), tensor.size(-1))

    def _attn(self, query, key, value, attention_mask, head_mask):
        attn_output, attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_output = attn_output / (key.size(-1) ** 0.5)

        if attention_mask is not None:
            attn_output = attn_output + attention_mask

        attn_weights = nn.functional.softmax(attn_output, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _merge_heads(self, tensor, num_heads, head_dim):
        tensor = tensor.view(-1, num_heads, tensor.size(-2), head_dim)
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(tensor.size(0), -1, num_heads * head_dim)
