# these imports are found in https://huggingface.co/microsoft/phi-1_5/blob/main/modeling_phi.py

import torch
from einops import rearrange
from typing import Optional
import math

pad_input, unpad_input = None, None
# FlashRotaryEmbedding = None
# FlashSelfAttention, FlashCrossAttention = None, None
# FusedDense = None

# -------------------------------------------------

# this overwrites the _forward_self_attn method, which is accessed by layer.mixer.
def new_forward_self_attn(
    self, x: torch.FloatTensor, key_padding_mask: Optional[torch.BoolTensor]
) -> torch.FloatTensor:
    print(f'forward pass in new _forward_self_attn! layer {self.layer_idx}')
    qkv = self.Wqkv(x)
    qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

    if self.rotary_dim > 0:
        qkv = self.rotary_emb(qkv)

    if self.flash_attn:
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]

        cu_seqlens, max_seqlen = None, None
        if key_padding_mask is not None:
            # If `key_padding_mask` is supplied, we need to unpad the input and retrieve
            # the `cu_seqlens` and `max_seqlen` to be used by `flash-attn`
            qkv, indices, cu_seqlens, max_seqlen = unpad_input(qkv, key_padding_mask)

        if self.checkpointing:
            attn_output = torch.utils.checkpoint.checkpoint(
                self.inner_attn, qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
        else:
            attn_output = self.inner_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).to(qkv.device)

        # If `key_padding_mask` is supplied, we need to pad the output back to the original shape
        return pad_input(attn_output, indices, batch_size, seqlen) if key_padding_mask is not None else attn_output

    if self.checkpointing:
        return torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, key_padding_mask=key_padding_mask)

    return self.inner_attn(qkv, key_padding_mask=key_padding_mask)

# this overwrites the forward method of self.inner_attn, which is defined in SelfAttention.
# it is accessed by layer.mixer.inner_attn.
def new_forward_inner_attn(
    self,
    qkv: torch.FloatTensor,
    causal: bool = None,
    key_padding_mask: Optional[torch.BoolTensor] = None,
    **kwargs,
) -> torch.FloatTensor:
    print('in new forward inner attn method!')
    batch_size, seqlen = qkv.shape[0], qkv.shape[1]
    q, k, v = qkv.unbind(dim=2)

    q = q.to(torch.float32)
    k = k.to(torch.float32)

    causal = self.causal if causal is None else causal
    softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

    # Autocast is manually disabled to avoid `torch.einsum` performing the operation
    # using float16, which might lead to overflow
    scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

    if key_padding_mask is not None:
        padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device)
        padding_mask.masked_fill_(key_padding_mask, 0.0)

        scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

    if causal:
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)

    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    attention = self.drop(attention)

    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output