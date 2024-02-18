from copy import deepcopy
from xml.dom.minidom import Identified
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
from lrp.layer_norm import LayerNorm as LRPLayerNorm

class LRPBackwardNoopModule(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, input1):
        return ActivationAlpha1Beta0.apply(self.module, input1)

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
    elif isinstance(nn_module, nn.LayerNorm):
        return LRPLayerNorm.from_torch(nn_module)
    elif isinstance(nn_module, (LRPResidual)):
        # nn_module.forward =
        # return nn.Identity

        return nn_module
    elif isinstance(nn_module, (nn.ReLU, nn.Dropout, nn.GELU)):

        return LRPBackwardNoopModule(nn_module)
    elif isinstance(nn_module, nn.Softmax):
        # return nn.Identity
        return LRPSoftmax.from_torch(nn_module)
    elif isinstance(nn_module, nn.ModuleList):
        return nn.ModuleList([ lrp_modification_module(module_list_item) for module_list_item in nn_module ])
    else:
        nested_modules = get_nested_nn_modules(nn_module)
        if len(nested_modules) > 0:
            # go recursively
            print("Going to convert recursively module", nn_module)
            convert_module(nn_module)
            return nn_module

        raise ValueError("Unsupported module for lrp modification:", nn_module)

def get_nested_nn_modules(model):
    nn_modules_list = []
    for attribute_name in dir(model):
        is_public = not attribute_name.startswith("_")
        attribute_value = getattr(model, attribute_name)
        is_nn_module = isinstance(attribute_value, nn.Module)
        if is_public and is_nn_module:
            nn_modules_list.append(attribute_name)
    return nn_modules_list


def convert_module(model):

    for attribute_name in get_nested_nn_modules(model):
        attribute_value = getattr(model, attribute_name)
        setattr(model, attribute_name, lrp_modification_module(attribute_value))

    return

def test_lrp_attention():

    from diffusers.models.attention_processor import Attention, AttnProcessor2_0

    inner_dim = 30
    attention = Attention(
        query_dim=inner_dim,
    )
    attention.eval()
    attention_original = deepcopy(attention)

    print("attention", attention)

    convert_module(attention)

    hidden_states = torch.rand([10, 7, inner_dim], requires_grad=True) # [bs, seq_len, hidden_dim]
    attention_out = attention.forward(hidden_states)
    attention_original_out = attention_original.forward(hidden_states)

    assert torch.allclose(attention_out, attention_original_out, atol=1e-6), 'blocks outputs are the same'

    print("attention_out", attention_out.shape)

    attention_out.backward( torch.ones_like(attention_out) / attention_out.numel() )

    hidden_states_grad_sum = hidden_states.grad.sum()
    print("hidden_states_grad_sum", hidden_states_grad_sum)

    assert torch.allclose(hidden_states_grad_sum, torch.tensor(1.), atol=1e-4), 'hidden_states_grad_sum is 1'

    return



def test_lrp_transformer_block():

    from diffusers.models.attention import BasicTransformerBlock

    inner_dim = 30
    transformer_block = BasicTransformerBlock(
        dim=inner_dim,
        num_attention_heads=3,
        attention_head_dim=10,
    )
    transformer_block.eval()
    transformer_block_original = deepcopy(transformer_block)

    print("transformer_block", transformer_block)

    convert_module(transformer_block)

    hidden_states = torch.rand([1, 3, inner_dim], requires_grad=True) # [bs, seq_len, hidden_dim]
    transformer_block_out = transformer_block.forward(hidden_states)
    transformer_block_original_out = transformer_block_original.forward(hidden_states)

    assert torch.allclose(transformer_block_out, transformer_block_original_out, atol=1e-6), 'blocks outputs are the same'

    print("transformer_block_out", transformer_block_out.shape)

    transformer_block_out.backward( torch.ones_like(transformer_block_out) / transformer_block_out.numel() )

    hidden_states_grad_sum = hidden_states.grad.sum()
    print("hidden_states_grad_sum", hidden_states_grad_sum)

    assert hidden_states_grad_sum == 1, 'hidden_states_grad_sum is 1'

    return

def test_lrp_mlp():
    m = IntegrationModule()
    convert_module(m)

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

