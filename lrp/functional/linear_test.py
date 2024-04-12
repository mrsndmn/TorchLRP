import pytest
import torch

from lrp.functional.linear import LinearAlpha1Beta0, LinearAlpha1Beta0Explicit

def test_lrp_linear_alpha_beta():

    in_features = 2
    out_features = 3

    weight = torch.rand([out_features, in_features], requires_grad=True)
    bias = torch.rand([out_features], requires_grad=True)


    batch_size = 1
    in_tensor = torch.rand([batch_size, in_features], requires_grad=True)

    result = LinearAlpha1Beta0.apply(in_tensor, weight, bias)

    loss = result.mean()
    loss.backward()

    implicit_in_tensor_grad = in_tensor.grad

    in_tensor.grad = None
    result = LinearAlpha1Beta0Explicit.apply(in_tensor, weight, bias)
    loss = result.mean()
    loss.backward()

    explicit_in_tensor_grad = in_tensor.grad

    assert torch.allclose(explicit_in_tensor_grad, implicit_in_tensor_grad, atol=1e-4), 'lrp grad is ok by both the methods'

