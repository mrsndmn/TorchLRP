import pytest
import torch

from lrp.functional.layer_norm import LayerNormAlpha1Beta0

def test_lrp_layer_norm_alpha_beta():

    in_features = 10
    num_heads = 7

    batch_size = 3
    in_tensor = torch.rand([batch_size, num_heads, in_features], requires_grad=True)

    normalized_shape = [in_features]
    weight = torch.rand(normalized_shape, requires_grad=True)
    bias = torch.rand(normalized_shape, requires_grad=True)

    print("weight", weight)

    result = LayerNormAlpha1Beta0.apply(in_tensor, normalized_shape, weight, bias)

    loss = result.mean()
    loss.backward()

    print("in_tensor.grad", in_tensor.grad)
    print("in_tensor.grad.sum", in_tensor.grad.sum())
    assert torch.allclose(in_tensor.grad.flatten(1).sum(dim=-1), torch.ones([batch_size]), atol=1e-4), 'sum relevance is constant'

