import pytest
import torch

from lrp.functional.layer_norm import LayerNormAlpha1Beta0

def test_lrp_layer_norm_alpha_beta():

    in_features = 2

    batch_size = 1
    in_tensor = torch.rand([batch_size, in_features], requires_grad=True)

    normalized_shape = [in_features]
    weight = torch.rand(normalized_shape, requires_grad=True)
    bias = torch.rand(normalized_shape, requires_grad=True)

    print("weight", weight)

    result = LayerNormAlpha1Beta0.apply(in_tensor, normalized_shape, weight, bias)

    loss = result.mean()
    loss.backward()

    print("in_tensor.grad", in_tensor.grad)
    assert torch.allclose(in_tensor.grad.sum(), torch.Tensor([1.]), atol=1e-4), 'sum relevance is constant'

