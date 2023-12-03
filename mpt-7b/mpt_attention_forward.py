import torch
import warnings
from einops import rearrange
import math
from typing import Optional, Tuple

# hacking this import for now. is located in attention.py of mpt-7b. not important for the actual pass.
def repeat_kv_for_gqa(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Perform repeat of kv heads along a particular dimension.
    hidden.shape expected to be: (batch size, seq len, kv_n_heads, head_dim)
    n_rep: amount of repetitions of kv_n_heads
    Unlike torch.repeat_interleave, this function avoids allocating new memory.
    """
    if n_rep == 1:
        return hidden
    (b, s, kv_n_heads, d) = hidden.shape
    hidden = hidden[:, :, :, None, :].expand(b, s, kv_n_heads, n_rep, d)
    return hidden.reshape(b, s, kv_n_heads * n_rep, d)


# this is a new forward method for the GroupedQueryAttention class.
def new_forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, attn_bias: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, is_causal: bool=True, needs_weights: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)
        (query, key, value) = qkv.split([self.d_model, self.kv_n_heads * self.head_dim, self.kv_n_heads * self.head_dim], dim=2)
        key_padding_mask = attention_mask
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)
        (context, attn_weights, past_key_value) = self.attn_fn(self, query, key, value, self.n_heads, self.kv_n_heads, past_key_value=past_key_value, softmax_scale=self.softmax_scale, attn_bias=attn_bias, key_padding_mask=key_padding_mask, is_causal=is_causal, dropout_p=self.attn_dropout_p, training=self.training, needs_weights=needs_weights)
        return (self.out_proj(context), attn_weights, past_key_value)


# this is the self.attn_fn function
def new_scaled_multihead_dot_product_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, n_heads: int, kv_n_heads: Optional[int]=None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, softmax_scale: Optional[float]=None, attn_bias: Optional[torch.Tensor]=None, key_padding_mask: Optional[torch.Tensor]=None, is_causal: bool=False, dropout_p: float=0.0, training: bool=False, needs_weights: bool=False, multiquery: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    if multiquery:
        warnings.warn(DeprecationWarning('The direct use of the multiquery arg is deprecated. Setting kv_n_heads=1 automatically. Please set kv_n_heads=1 explicitly to remove this warning.'))
        kv_n_heads = 1
    elif kv_n_heads is None:
        warnings.warn(DeprecationWarning('Not specifying a value for the kv_n_heads arg is deprecated. Setting kv_n_heads=n_heads automatically. Please set kv_n_heads=n_heads explicitly to remove this warning.'))
        kv_n_heads = n_heads
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0], k], dim=3)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v)
    (b, _, s_q, d) = q.shape
    s_k = k.size(-1)
    if kv_n_heads > 1 and kv_n_heads < n_heads:
        k = repeat_kv_for_gqa(k.transpose(1, 2), n_heads // kv_n_heads).transpose(1, 2)
        v = repeat_kv_for_gqa(v.transpose(1, 2), n_heads // kv_n_heads).transpose(1, 2)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)
    attn_weight = q.matmul(k) * softmax_scale
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
        if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
            raise RuntimeError(f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
        attn_weight = attn_weight + attn_bias
    min_val = torch.finfo(q.dtype).min
    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn('Propagating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unnecessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
        attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1, 1, s_k)), min_val)
    if is_causal and (not q.size(2) == 1):
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float32)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
    attn_weight = torch.softmax(attn_weight, dim=-1) # size (batch, heads, seqlen, seqlen)
    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=training, inplace=True)

    # we want to use attn_weight here.
    # print(f'attn_weight:{attn_weight.shape}')
    # torch.sum(attn_weight, dim=-1) -> all 1's
    intermediate_entropy = torch.nan_to_num(attn_weight * torch.log(attn_weight), nan=0.0)
    entropy = -torch.sum(intermediate_entropy, dim=-1) # size (batch, heads, seqlen)
    last_n = 20
    last_n_entropy = entropy[..., -20:]
    self.last_n_entropy = last_n_entropy
    
    out = attn_weight.to(v.dtype).matmul(v) # out is of size (batch, heads, seqlen, hiddendim)
    out = rearrange(out, 'b h s d -> b s (h d)') # out is size (batch, seqlen, heads*hiddendim) after this
    if needs_weights:
        return (out, attn_weight, past_key_value)
    return (out, None, past_key_value)