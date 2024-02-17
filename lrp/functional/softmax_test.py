import pytest
import torch

from lrp.functional.softmax import SoftmaxAlpha1Beta0

def test_lrp_softmax_alpha_beta():

    in_features = 10

    batch_size = 1
    in_tensor = torch.rand([batch_size, in_features], requires_grad=True)

    result = SoftmaxAlpha1Beta0.apply(in_tensor, -1)

    loss = result.mean()
    loss.backward()

    print("in_tensor.grad", in_tensor.grad)
    assert torch.allclose(in_tensor.grad.sum(), torch.Tensor([1.]), atol=1e-4), 'sum relevance is constant'

