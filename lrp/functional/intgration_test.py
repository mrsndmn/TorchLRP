import pytest
import torch
import torch.nn as nn
from torch.autograd import Function

from lrp.functional.residual import ResidualAlpha1Beta0
from lrp.functional.softmax import SoftmaxAlpha1Beta0
from lrp.functional.linear import LinearAlpha1Beta0
from lrp.functional.layer_norm import LayerNormAlpha1Beta0

from lrp.linear import Linear as LRPLinear
from lrp.softmax import Softmax as LRPSoftmax
from lrp.residual import Residual as LRPResidual

class LRPActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()

        self.activation = activation

    def forward(self, input1):
        return ActivationAlpha1Beta0.apply(self.activation, input1)

class ActivationAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, activation_module, input1):
        return activation_module(input1)

    @staticmethod
    def backward(ctx, relevance_output):
        return None, relevance_output


class IntegrationModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = nn.ReLU()
        self.lin1 = nn.Linear(20, 40)
        self.lin2 = nn.Linear(40, 40)
        self.out = nn.Linear(40, 10)

        self.softmax = nn.Softmax(dim=-1)

        self.resudual = LRPResidual()

        return

    def forward(self, input):

        latent1 = self.activation(self.lin1(input))
        latent2 = self.activation(self.lin2(latent1))
        output = self.out(self.resudual(latent1, latent2))

        return self.softmax(output)

def lrp_modification_module(nn_module):
    print("lrp modify", nn_module)

    if isinstance(nn_module, nn.Linear):
        return LRPLinear.from_torch(nn_module)
    elif isinstance(nn_module, (LRPResidual)):
        # nn_module.forward =
        # return nn.Identity
        return nn_module
    elif isinstance(nn_module, nn.ReLU):
        return LRPActivation(nn_module)
    elif isinstance(nn_module, nn.Softmax):
        return LRPSoftmax.from_torch(nn_module)
    else:
        raise ValueError("Unsupported module for lrp modification:", nn_module)

def convert_module(model):

    for attribute in dir(model):
        is_public = not attribute.startswith("_")
        attribute_value = getattr(model, attribute)
        is_nn_module = isinstance(attribute_value, nn.Module)

        if is_public and is_nn_module:
            setattr(model, attribute, lrp_modification_module(attribute_value))

    return

def test_lrp_mlp():
    m = IntegrationModule()
    convert_module(m)

    import random
    random.seed(0)
    torch.manual_seed(1)
    import numpy as np
    np.random.seed(0)

    bs = 1
    in_features = m.lin1.in_features
    input = torch.rand([bs, in_features], requires_grad=True)

    print("input sum", input.sum(), "mean", input.mean(), "min", input.min(), "max", input.max())

    latent1 = m.activation(m.lin1(input))
    latent2 = m.activation(m.lin2(latent1))
    output = m.out(m.resudual(latent1, latent2))
    probas = m.softmax(output)

    probas.backward(torch.ones_like(probas))

    input_grad_sum = input.grad.sum(dim=-1)
    print(input_grad_sum)

    assert torch.allclose(input_grad_sum, torch.ones_like(input_grad_sum), atol=1e-3), 'relevance amount did not changed'

