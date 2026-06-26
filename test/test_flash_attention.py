import torch
import pytest

from src.backend.flash_attention import (
    MaskedFlashAttention,
    masked_flash_attention,
    masked_flash_attention_backward,
    masked_flash_attention_forward,
    torch_masked_attention,
)

try:
    from src.backend.flash_attention_triton import triton_available, triton_masked_flash_attention
except Exception:  # pragma: no cover - defensive import guard for CPU-only envs.
    triton_available = lambda: False
    triton_masked_flash_attention = None


def test_masked_flash_attention_forward_matches_torch_reference():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 7, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    segment_ids = torch.tensor(
        [
            [1, 1, 1, 2, 2, 0, 0],
            [1, 1, 2, 2, 3, 3, 0],
        ]
    )

    expected = torch_masked_attention(q, k, v, segment_ids)
    actual = masked_flash_attention_forward(q, k, v, segment_ids, block_m=3, block_n=2)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_masked_flash_attention_backward_matches_torch_reference():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 6, 4, requires_grad=True)
    k = torch.randn(1, 2, 6, 4, requires_grad=True)
    v = torch.randn(1, 2, 6, 4, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    segment_ids = torch.tensor([[1, 1, 2, 2, 2, 0]])
    grad = torch.randn_like(q)

    actual = masked_flash_attention(q, k, v, segment_ids, block_m=2, block_n=3)
    expected = torch_masked_attention(q_ref, k_ref, v_ref, segment_ids)
    actual.backward(grad)
    expected.backward(grad)

    torch.testing.assert_close(q.grad, q_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=1e-5, rtol=1e-5)


def test_masked_flash_attention_backward_wrapper_returns_reference_grads():
    torch.manual_seed(3)
    q = torch.randn(1, 2, 5, 4, requires_grad=True)
    k = torch.randn(1, 2, 5, 4, requires_grad=True)
    v = torch.randn(1, 2, 5, 4, requires_grad=True)
    segment_ids = torch.tensor([[1, 1, 2, 2, 0]])
    grad = torch.randn_like(q)

    expected = torch_masked_attention(q, k, v, segment_ids)
    expected.backward(grad)
    grad_q, grad_k, grad_v = masked_flash_attention_backward(
        q.detach(),
        k.detach(),
        v.detach(),
        segment_ids,
        grad,
    )

    torch.testing.assert_close(grad_q, q.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_k, k.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_v, v.grad, atol=1e-5, rtol=1e-5)


def test_masked_flash_attention_module_matches_wrapper():
    torch.manual_seed(2)
    q = torch.randn(1, 1, 5, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    segment_ids = torch.tensor([[1, 1, 1, 0, 0]])
    module = MaskedFlashAttention(block_m=2, block_n=2)

    expected = masked_flash_attention(q, k, v, segment_ids, block_m=2, block_n=2)
    actual = module(q, k, v, segment_ids)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="Triton CUDA is unavailable")
def test_triton_masked_flash_attention_forward_matches_torch_reference_on_cuda():
    torch.manual_seed(4)
    q = torch.randn(2, 3, 32, 16, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    segment_ids = torch.tensor(
        [
            [1] * 12 + [2] * 12 + [0] * 8,
            [1] * 16 + [2] * 16,
        ],
        device="cuda",
    )

    expected = torch_masked_attention(q, k, v, segment_ids)
    actual = triton_masked_flash_attention(q, k, v, segment_ids, block_m=8, block_n=16)

    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available() or not triton_available(), reason="Triton CUDA is unavailable")
def test_triton_masked_flash_attention_backward_matches_torch_reference_on_cuda():
    torch.manual_seed(5)
    q = torch.randn(1, 2, 24, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    segment_ids = torch.tensor([[1] * 10 + [2] * 10 + [0] * 4], device="cuda")
    grad = torch.randn_like(q)

    actual = triton_masked_flash_attention(q, k, v, segment_ids, block_m=8, block_n=16)
    expected = torch_masked_attention(q_ref, k_ref, v_ref, segment_ids)
    actual.backward(grad)
    expected.backward(grad)

    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-4, rtol=5e-4)
