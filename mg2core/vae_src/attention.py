"""Attention for the vendored Matrix-Game 2.0 code: PyTorch scaled_dot_product_attention only.

The official code imports flash-attn (FA2/FA3) unconditionally in the action module and in the
cross-attention. Workers built from a comfy-build spec run a current torch with no flash-attn
wheel, so every entry point here maps onto torch.nn.functional.scaled_dot_product_attention,
which picks the flash / memory-efficient kernel that is available on the device.
Layouts follow flash-attn: q [B, Lq, H, D], k/v [B, Lk, H, D] -> [B, Lq, H, D].
"""
import torch

FLASH_ATTN_2_AVAILABLE = False
FLASH_ATTN_3_AVAILABLE = False

__all__ = ["flash_attn_func", "flash_attention", "attention"]


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_ignored):
    dt = q.dtype
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.to(dt).transpose(1, 2), v.to(dt).transpose(1, 2),
        is_causal=causal, dropout_p=dropout_p, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
                    causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16,
                    version=None):
    out_dtype = q.dtype
    if q_scale is not None:
        q = q * q_scale
    out = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
    return out.type(out_dtype)


def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
              causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, fa_version=None):
    return flash_attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale, causal, window_size,
                           deterministic, dtype)
