import pytest
import torch

from lrp.functional.softmax import SoftmaxAlpha1Beta0

def test_lrp_softmax_alpha_beta():

    in_features = 10
    num_heads = 3
    batch_size = 2
    in_tensor = torch.rand([batch_size, num_heads, in_features], requires_grad=True)

    result = SoftmaxAlpha1Beta0.apply(in_tensor, -1)

    result.backward( torch.ones_like(result) / result.numel() )

    print("in_tensor.grad", in_tensor.grad)
    assert torch.allclose(in_tensor.grad.sum(), torch.Tensor([1.]), atol=1e-4), 'sum relevance is constant'

