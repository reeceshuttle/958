# these imports are found in https://huggingface.co/microsoft/phi-1_5/blob/main/modeling_phi.py

import torch
from einops import rearrange
from typing import Optional
import math

pad_input, unpad_input = None, None

# this overwrites the forward method of self.inner_attn, which is defined in SelfAttention.
# it is accessed by layer.mixer.inner_attn.
def new_forward_inner_attn(
    self,
    qkv: torch.FloatTensor,
    causal: bool = None,
    key_padding_mask: Optional[torch.BoolTensor] = None,
    **kwargs,
) -> torch.FloatTensor:
    batch_size, seqlen = qkv.shape[0], qkv.shape[1]
    q, k, v = qkv.unbind(dim=2)
    # q is shape (batch, seqlen, heads, headdim)
    q = q.to(torch.float32)
    k = k.to(torch.float32)

    causal = self.causal if causal is None else causal
    softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

    # Autocast is manually disabled to avoid `torch.einsum` performing the operation
    # using float16, which might lead to overflow
    scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    # key_padding_mask is None
    if key_padding_mask is not None:
        padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device)
        padding_mask.masked_fill_(key_padding_mask, 0.0)

        scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
    # casual is True
    if causal:
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)

    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # attention is shape (batch, heads, seqlen, seqlen)
    #                    (    1,    32,     81,     81)
    # summing attention across dim=-1 leads to all values of 1.
    # since nan implies log(0) was done and the limit xlogx goes to zero, we replace nan w 0.
    intermediate_entropy = torch.nan_to_num(attention * torch.log(attention), nan=0.0)
    entropy = -torch.sum(intermediate_entropy, dim=-1) # of shape (batch, heads, seqlen)
    last_n = 20
    last_n_entropy = entropy[..., -20:]
    # print(f'entropy shape:{entropy.shape}, last_{last_n}_entropy shape:{last_n_entropy.shape}')
    self.last_n_entropy = last_n_entropy

    attention = self.drop(attention)

    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output