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

    if key_padding_mask is not None:
        padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device)
        padding_mask.masked_fill_(key_padding_mask, 0.0)

        scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

    if causal:
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)

    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # attention is shape (batch, heads, seqlen, seqlen)
    #                    (    1,    32,     81,     81)
    # summing attention across dim=-1 leads to all values of 1.
    # since nan implies log(0) was done and the limit xlogx goes to zero, we replace nan w 0.
    intermediate_entropy = torch.nan_to_num(attention * torch.log(attention), nan=0.0)
    # entropy is of shape (batch, heads, seqlen)
    entropy = -torch.sum(intermediate_entropy, dim=-1)
    # print(entropy[0][0])
    avg_entropy_withinheads = torch.mean(entropy, dim=-1)
    # avg_entropy is of shape (1, 32)
    self.avg_entropy = avg_entropy_withinheads
    # averaging the entropy within heads due to memory issues: might do stats analysis in here later?
    # later: convert to new dtype to save memory? it is float32 by default

    attention = self.drop(attention)

    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output