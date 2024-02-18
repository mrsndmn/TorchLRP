import pytest
import torch

from lrp.functional.matmul import MatMulAlpha1Beta0

def test_lrp_matmul_alpha_beta():

    batch_size = 3
    num_heads = 8
    source_seq_len = 2
    hidden_dim = 4
    target_seq_len = 3

    input1 = torch.rand([batch_size, 8, 3, 3], requires_grad=True)
    input2 = torch.rand([batch_size, 8, 3, 64], requires_grad=True)

    # torch.Size([1, 8, 3, 64]) torch.Size([1, 8, 64, 3])
    print("input1.shape", input1.shape)
    print("input2.shape", input2.shape)

    result = MatMulAlpha1Beta0.apply(input1, input2)

    result.backward(torch.ones_like(result) / result.numel())

    print('input1.grad.sum()', input1.grad.sum())
    print('input2.grad.sum()', input2.grad.sum())

    assert torch.allclose(input1.grad.sum() + input2.grad.sum(), torch.tensor(1.0), atol=1e-4), 'all close 1 relevance'


