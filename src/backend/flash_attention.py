from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
from torch import nn


def masked_attention_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    seq_len = segment_ids.size(1)
    positions = torch.arange(seq_len, device=segment_ids.device)
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    causal = positions[:, None] >= positions[None, :]
    non_pad = segment_ids[:, :, None] != 0
    return same_segment & causal & non_pad


def torch_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    segment_ids: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    scale = scale or q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = masked_attention_mask(segment_ids).unsqueeze(1)
    scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    return torch.matmul(weights, v) * (segment_ids[:, None, :, None] != 0)


def masked_flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    segment_ids: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    scale: float | None = None,
) -> torch.Tensor:
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
    scale = scale or q.size(-1) ** -0.5

    batch_size, n_heads, seq_len, head_dim = q.shape
    output = torch.zeros_like(q)
    positions = torch.arange(seq_len, device=q.device)

    for q_start in range(0, seq_len, block_m):
        q_end = min(q_start + block_m, seq_len)
        q_block = q[:, :, q_start:q_end, :]
        q_segments = segment_ids[:, q_start:q_end]
        q_positions = positions[q_start:q_end]
        block_size = q_end - q_start

        acc = torch.zeros(batch_size, n_heads, block_size, head_dim, device=q.device, dtype=torch.float32)
        row_max = torch.full(
            (batch_size, n_heads, block_size),
            -torch.inf,
            device=q.device,
            dtype=torch.float32,
        )
        row_sum = torch.zeros(batch_size, n_heads, block_size, device=q.device, dtype=torch.float32)

        for k_start in range(0, seq_len, block_n):
            k_end = min(k_start + block_n, seq_len)
            k_segments = segment_ids[:, k_start:k_end]
            k_positions = positions[k_start:k_end]
            valid = (
                (q_segments[:, :, None] == k_segments[:, None, :])
                & (q_segments[:, :, None] != 0)
                & (q_positions[:, None] >= k_positions[None, :])
            )
            if not bool(valid.any()):
                continue

            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]
            scores = torch.matmul(q_block.float(), k_block.float().transpose(-2, -1)) * scale
            scores = scores.masked_fill(~valid[:, None, :, :], -torch.inf)
            block_max = scores.max(dim=-1).values
            has_values = valid.any(dim=-1)[:, None, :]
            block_max = torch.where(has_values, block_max, row_max)
            new_row_max = torch.maximum(row_max, block_max)

            old_scale = torch.where(
                torch.isfinite(row_max),
                torch.exp(row_max - new_row_max),
                torch.zeros_like(row_max),
            )
            normalized_scores = scores - new_row_max[:, :, :, None]
            normalized_scores = torch.where(
                valid[:, None, :, :],
                normalized_scores,
                torch.full_like(normalized_scores, -torch.inf),
            )
            probs = torch.exp(normalized_scores)
            acc = acc * old_scale[:, :, :, None] + torch.matmul(probs, v_block.float())
            row_sum = row_sum * old_scale + probs.sum(dim=-1)
            row_max = new_row_max

        safe_sum = row_sum.clamp_min(torch.finfo(row_sum.dtype).tiny)
        output[:, :, q_start:q_end, :] = (acc / safe_sum[:, :, :, None]).to(output.dtype)

    return output * (segment_ids[:, None, :, None] != 0)


def masked_flash_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    segment_ids: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = scale or q.size(-1) ** -0.5
    mask = masked_attention_mask(segment_ids).unsqueeze(1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(mask, weights, torch.zeros_like(weights))

    grad_v = torch.matmul(weights.transpose(-2, -1), grad_output)
    grad_weights = torch.matmul(grad_output, v.transpose(-2, -1))
    correction = (grad_weights * weights).sum(dim=-1, keepdim=True)
    grad_scores = weights * (grad_weights - correction)
    grad_scores = torch.where(mask, grad_scores, torch.zeros_like(grad_scores))
    grad_q = torch.matmul(grad_scores, k) * scale
    grad_k = torch.matmul(grad_scores.transpose(-2, -1), q) * scale
    return grad_q, grad_k, grad_v


class MaskedFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        segment_ids: torch.Tensor,
        block_m: int = 64,
        block_n: int = 64,
        scale: float | None = None,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v, segment_ids)
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.scale = scale
        return masked_flash_attention_forward(q, k, v, segment_ids, block_m, block_n, scale)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, segment_ids = ctx.saved_tensors
        grad_q, grad_k, grad_v = masked_flash_attention_backward(q, k, v, segment_ids, grad_output, ctx.scale)
        return grad_q, grad_k, grad_v, None, None, None, None


def masked_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    segment_ids: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    scale: float | None = None,
) -> torch.Tensor:
    return MaskedFlashAttentionFunction.apply(q, k, v, segment_ids, block_m, block_n, scale)


class MaskedFlashAttention(nn.Module):
    def __init__(self, block_m: int = 64, block_n: int = 64, scale: float | None = None) -> None:
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
        return masked_flash_attention(q, k, v, segment_ids, self.block_m, self.block_n, self.scale)


@dataclass(frozen=True)
class BenchmarkResult:
    torch_ms: float
    flash_ms: float
    speedup: float
    torch_score_memory_mb: float
    flash_score_memory_mb: float
    memory_ratio: float


def _time_call(fn, repeats: int) -> float:
    values: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(values)


def benchmark_flash_attention(
    batch_size: int = 2,
    n_heads: int = 4,
    seq_len: int = 512,
    head_dim: int = 64,
    block_m: int = 64,
    block_n: int = 64,
    repeats: int = 10,
    device: str | None = None,
    backend: str = "torch-blocked",
) -> BenchmarkResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    segment_ids = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)

    torch_masked_attention(q, k, v, segment_ids)
    if backend == "torch-blocked":
        flash_fn = lambda: masked_flash_attention_forward(q, k, v, segment_ids, block_m, block_n)
    elif backend == "triton":
        from src.backend.flash_attention_triton import triton_masked_flash_attention

        flash_fn = lambda: triton_masked_flash_attention(q, k, v, segment_ids, block_m, block_n)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    flash_fn()

    torch_ms = _time_call(lambda: torch_masked_attention(q, k, v, segment_ids), repeats)
    flash_ms = _time_call(flash_fn, repeats)

    element_size = q.element_size()
    torch_memory = batch_size * n_heads * seq_len * seq_len * element_size / (1024**2)
    flash_memory = batch_size * n_heads * block_m * block_n * element_size / (1024**2)
    return BenchmarkResult(
        torch_ms=torch_ms,
        flash_ms=flash_ms,
        speedup=torch_ms / flash_ms if flash_ms > 0 else float("inf"),
        torch_score_memory_mb=torch_memory,
        flash_score_memory_mb=flash_memory,
        memory_ratio=torch_memory / flash_memory if flash_memory > 0 else float("inf"),
    )


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark masked FlashAttention against torch attention")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device")
    parser.add_argument("--backend", choices=["torch-blocked", "triton"], default="torch-blocked")
    return parser


def run_benchmark_from_args(args: argparse.Namespace) -> BenchmarkResult:
    return benchmark_flash_attention(
        batch_size=args.batch_size,
        n_heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        block_m=args.block_m,
        block_n=args.block_n,
        repeats=args.repeats,
        device=args.device,
        backend=args.backend,
    )
