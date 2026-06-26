from __future__ import annotations

import torch
from torch import nn


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on machines without Triton.
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None


def _require_triton_cuda(q: torch.Tensor) -> None:
    if triton is None or tl is None:
        raise RuntimeError("Triton is not installed. In Colab run: pip install triton")
    if not q.is_cuda:
        raise RuntimeError("Triton FlashAttention requires CUDA tensors.")


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


if triton is not None:

    @triton.jit
    def _masked_flash_attention_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        segment_ptr,
        out_ptr,
        scale: tl.constexpr,
        n_heads: tl.constexpr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        stride_bh: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)
        batch_id = pid_bh // n_heads
        q_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offsets = tl.arange(0, BLOCK_D)

        q = tl.load(
            q_ptr + pid_bh * stride_bh + q_offsets[:, None] * head_dim + d_offsets[None, :],
            mask=(q_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        q_segments = tl.load(
            segment_ptr + batch_id * seq_len + q_offsets,
            mask=q_offsets < seq_len,
            other=0,
        )

        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for k_start in range(0, seq_len, BLOCK_N):
            k_offsets = k_start + tl.arange(0, BLOCK_N)
            k_segments = tl.load(
                segment_ptr + batch_id * seq_len + k_offsets,
                mask=k_offsets < seq_len,
                other=0,
            )
            k = tl.load(
                k_ptr + pid_bh * stride_bh + k_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=(k_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim),
                other=0.0,
            )
            v = tl.load(
                v_ptr + pid_bh * stride_bh + k_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=(k_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim),
                other=0.0,
            )
            valid = (
                (q_offsets[:, None] < seq_len)
                & (k_offsets[None, :] < seq_len)
                & (q_segments[:, None] != 0)
                & (q_segments[:, None] == k_segments[None, :])
                & (q_offsets[:, None] >= k_offsets[None, :])
            )
            scores = tl.dot(q, tl.trans(k)) * scale
            scores = tl.where(valid, scores, -float("inf"))
            block_max = tl.max(scores, axis=1)
            new_row_max = tl.maximum(row_max, block_max)
            safe_new_row_max = tl.where(new_row_max == -float("inf"), 0.0, new_row_max)
            old_scale = tl.where(row_max == -float("inf"), 0.0, tl.exp(row_max - safe_new_row_max))
            probs = tl.exp(scores - safe_new_row_max[:, None])
            probs = tl.where(valid, probs, 0.0)
            acc = acc * old_scale[:, None] + tl.dot(probs.to(v.dtype), v)
            row_sum = row_sum * old_scale + tl.sum(probs, axis=1)
            row_max = new_row_max

        safe_row_sum = tl.maximum(row_sum, 1.0e-20)
        out = acc / safe_row_sum[:, None]
        out = tl.where((q_segments[:, None] != 0) & (d_offsets[None, :] < head_dim), out, 0.0)
        tl.store(
            out_ptr + pid_bh * stride_bh + q_offsets[:, None] * head_dim + d_offsets[None, :],
            out,
            mask=(q_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim),
        )

    @triton.jit
    def _masked_flash_attention_backward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        segment_ptr,
        grad_out_ptr,
        grad_q_ptr,
        grad_k_ptr,
        grad_v_ptr,
        scale: tl.constexpr,
        n_heads: tl.constexpr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        stride_bh: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        row = tl.program_id(1)
        batch_id = pid_bh // n_heads
        d_offsets = tl.arange(0, BLOCK_D)

        q = tl.load(
            q_ptr + pid_bh * stride_bh + row * head_dim + d_offsets,
            mask=d_offsets < head_dim,
            other=0.0,
        )
        grad_out = tl.load(
            grad_out_ptr + pid_bh * stride_bh + row * head_dim + d_offsets,
            mask=d_offsets < head_dim,
            other=0.0,
        )
        q_segment = tl.load(segment_ptr + batch_id * seq_len + row)

        row_max = tl.full((), -float("inf"), tl.float32)
        row_sum = tl.full((), 0.0, tl.float32)
        for k_start in range(0, seq_len, BLOCK_N):
            k_offsets = k_start + tl.arange(0, BLOCK_N)
            k_segments = tl.load(
                segment_ptr + batch_id * seq_len + k_offsets,
                mask=k_offsets < seq_len,
                other=0,
            )
            k = tl.load(
                k_ptr + pid_bh * stride_bh + k_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=(k_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim),
                other=0.0,
            )
            valid = (
                (k_offsets < seq_len)
                & (q_segment != 0)
                & (q_segment == k_segments)
                & (row >= k_offsets)
            )
            scores = tl.sum(k * q[None, :], axis=1) * scale
            scores = tl.where(valid, scores, -float("inf"))
            block_max = tl.max(scores, axis=0)
            new_row_max = tl.maximum(row_max, block_max)
            safe_new_row_max = tl.where(new_row_max == -float("inf"), 0.0, new_row_max)
            old_scale = tl.where(row_max == -float("inf"), 0.0, tl.exp(row_max - safe_new_row_max))
            probs = tl.exp(scores - safe_new_row_max)
            probs = tl.where(valid, probs, 0.0)
            row_sum = row_sum * old_scale + tl.sum(probs, axis=0)
            row_max = new_row_max

        safe_row_max = tl.where(row_max == -float("inf"), 0.0, row_max)
        safe_row_sum = tl.maximum(row_sum, 1.0e-20)
        grad_q = tl.zeros((BLOCK_D,), tl.float32)
        correction = tl.full((), 0.0, tl.float32)
        for k_start in range(0, seq_len, BLOCK_N):
            k_offsets = k_start + tl.arange(0, BLOCK_N)
            k_segments = tl.load(
                segment_ptr + batch_id * seq_len + k_offsets,
                mask=k_offsets < seq_len,
                other=0,
            )
            k = tl.load(
                k_ptr + pid_bh * stride_bh + k_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=(k_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim),
                other=0.0,
            )
            v = tl.load(
                v_ptr + pid_bh * stride_bh + k_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=(k_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim),
                other=0.0,
            )
            valid = (
                (k_offsets < seq_len)
                & (q_segment != 0)
                & (q_segment == k_segments)
                & (row >= k_offsets)
            )
            scores = tl.sum(k * q[None, :], axis=1) * scale
            probs = tl.exp(scores - safe_row_max) / safe_row_sum
            probs = tl.where(valid, probs, 0.0)
            grad_probs = tl.sum(v * grad_out[None, :], axis=1)
            correction += tl.sum(probs * grad_probs, axis=0)

        for k_start in range(0, seq_len, BLOCK_N):
            k_offsets = k_start + tl.arange(0, BLOCK_N)
            k_segments = tl.load(
                segment_ptr + batch_id * seq_len + k_offsets,
                mask=k_offsets < seq_len,
                other=0,
            )
            k = tl.load(
                k_ptr + pid_bh * stride_bh + k_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=(k_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim),
                other=0.0,
            )
            v = tl.load(
                v_ptr + pid_bh * stride_bh + k_offsets[:, None] * head_dim + d_offsets[None, :],
                mask=(k_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim),
                other=0.0,
            )
            valid = (
                (k_offsets < seq_len)
                & (q_segment != 0)
                & (q_segment == k_segments)
                & (row >= k_offsets)
            )
            scores = tl.sum(k * q[None, :], axis=1) * scale
            probs = tl.exp(scores - safe_row_max) / safe_row_sum
            probs = tl.where(valid, probs, 0.0)
            grad_probs = tl.sum(v * grad_out[None, :], axis=1)
            grad_scores = probs * (grad_probs - correction)

            tl.atomic_add(
                grad_v_ptr + pid_bh * stride_bh + k_offsets[:, None] * head_dim + d_offsets[None, :],
                probs[:, None] * grad_out[None, :],
                sem="relaxed",
                mask=(k_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim) & valid[:, None],
            )
            tl.atomic_add(
                grad_k_ptr + pid_bh * stride_bh + k_offsets[:, None] * head_dim + d_offsets[None, :],
                grad_scores[:, None] * q[None, :] * scale,
                sem="relaxed",
                mask=(k_offsets[:, None] < seq_len) & (d_offsets[None, :] < head_dim) & valid[:, None],
            )
            grad_q += tl.sum(grad_scores[:, None] * k * scale, axis=0)

        grad_q = tl.where(q_segment != 0, grad_q, 0.0)
        tl.store(
            grad_q_ptr + pid_bh * stride_bh + row * head_dim + d_offsets,
            grad_q,
            mask=d_offsets < head_dim,
        )


class TritonMaskedFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        segment_ids: torch.Tensor,
        block_m: int = 16,
        block_n: int = 64,
        scale: float | None = None,
    ) -> torch.Tensor:
        _require_triton_cuda(q)
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k and v must have the same shape [batch, heads, seq_len, head_dim]")
        if q.ndim != 4:
            raise ValueError("q, k and v must be 4D tensors: [batch, heads, seq_len, head_dim]")
        if segment_ids.shape != (q.size(0), q.size(2)):
            raise ValueError("segment_ids must have shape [batch, seq_len]")
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        segment_ids = segment_ids.contiguous()
        batch_size, n_heads, seq_len, head_dim = q.shape
        block_d = _next_power_of_2(head_dim)
        if block_d > 128:
            raise ValueError("Triton backend supports head_dim <= 128")
        scale = scale or head_dim**-0.5
        output = torch.empty_like(q)
        grid = (batch_size * n_heads, triton.cdiv(seq_len, block_m))
        _masked_flash_attention_forward_kernel[grid](
            q,
            k,
            v,
            segment_ids,
            output,
            scale,
            n_heads,
            seq_len,
            head_dim,
            seq_len * head_dim,
            block_m,
            block_n,
            block_d,
            num_warps=4,
        )
        ctx.save_for_backward(q, k, v, segment_ids)
        ctx.block_n = block_n
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, segment_ids = ctx.saved_tensors
        batch_size, n_heads, seq_len, head_dim = q.shape
        block_d = _next_power_of_2(head_dim)
        grad_q = torch.empty_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        grid = (batch_size * n_heads, seq_len)
        _masked_flash_attention_backward_kernel[grid](
            q,
            k,
            v,
            segment_ids,
            grad_output.contiguous(),
            grad_q,
            grad_k,
            grad_v,
            ctx.scale,
            n_heads,
            seq_len,
            head_dim,
            seq_len * head_dim,
            ctx.block_n,
            block_d,
            num_warps=4,
        )
        return grad_q, grad_k, grad_v, None, None, None, None


def triton_masked_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    segment_ids: torch.Tensor,
    block_m: int = 16,
    block_n: int = 64,
    scale: float | None = None,
) -> torch.Tensor:
    return TritonMaskedFlashAttentionFunction.apply(q, k, v, segment_ids, block_m, block_n, scale)


class TritonMaskedFlashAttention(nn.Module):
    def __init__(self, block_m: int = 16, block_n: int = 64, scale: float | None = None) -> None:
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.scale = scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        return triton_masked_flash_attention(q, k, v, segment_ids, self.block_m, self.block_n, self.scale)
