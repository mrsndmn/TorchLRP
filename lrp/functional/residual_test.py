import pytest
import torch

from lrp.functional.residual import ResidualAlpha1Beta0

def test_lrp_residual_alpha_beta():

    in_features = 2
    batch_size = 1
    in_tensor1 = torch.rand([batch_size, in_features], requires_grad=True)
    in_tensor2 = torch.rand([batch_size, in_features], requires_grad=True)

    result = ResidualAlpha1Beta0.apply(in_tensor1, in_tensor2)

    loss = result.mean()
    loss.backward()

    print("in_tensor.grad", in_tensor1.grad)
    print("in_tensor.grad", in_tensor2.grad)
    assert torch.allclose(in_tensor1.grad.sum() + in_tensor2.grad.sum(), torch.Tensor([1.]), atol=1e-4), 'sum relevance is constant'

