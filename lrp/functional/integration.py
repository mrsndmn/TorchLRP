from copy import deepcopy
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from lrp.linear import Linear as LRPLinear
from lrp.mul import Mul as LRPMul
from lrp.matmul import MatMul as LRPMatMul
from lrp.mul_const import MulConst as LRPMulConst
from lrp.add_const import AddConst as LRPAddConst
from lrp.softmax import Softmax as LRPSoftmax
from lrp.residual import Residual as LRPResidual
from lrp.layer_norm import LayerNorm as LRPLayerNorm

from diffusers.models.attention import GEGLU

class LRPBackwardNoopModule(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, input1):
        return NoopAlpha1Beta0.apply(self.module, input1)

class NoopAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, activation_module, input1):
        return activation_module(input1)

    @staticmethod
    def backward(ctx, relevance_output):
        return None, relevance_output


class LRPBackwardZeroGradModule(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, input1):
        return NoopAlpha1Beta0.apply(self.module, input1)

class ZeroGradAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, module, input1):
        return module(input1)

    @staticmethod
    def backward(ctx, relevance_output):
        return None, torch.zeros_like(relevance_output)


def lrp_modification_module(nn_module, rule):
    print("lrp modify", nn_module)

    if isinstance(nn_module, nn.Linear):
        return LRPLinear.from_torch(nn_module)
    elif isinstance(nn_module, nn.LayerNorm):
        return LRPLayerNorm.from_torch(nn_module)
    elif isinstance(nn_module, (LRPResidual, LRPMul, LRPMulConst, LRPMatMul, LRPAddConst)):
        nn_module.rule = rule
        return nn_module
    elif isinstance(nn_module, (nn.Embedding)):
        # nn_module.forward =
        # return nn.Identity
        return nn_module
    elif isinstance(nn_module, (nn.ReLU, nn.Dropout, nn.SiLU, nn.Sigmoid, nn.GELU)):

        return LRPBackwardNoopModule(nn_module)
    elif isinstance(nn_module, nn.Softmax):
        # return nn.Identity
        return LRPSoftmax.from_torch(nn_module)
    elif isinstance(nn_module, nn.ModuleList):
        return nn.ModuleList([ lrp_modification_module(module_list_item, rule=rule) for module_list_item in nn_module ])
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

EXTRA_TORCH_TENSOR_OPERATION_MAP = {
    'tensor_log': partial(NoopAlpha1Beta0.apply, torch.log),
    # 'gelu': partial(NoopAlpha1Beta0.apply, F.gelu),
}

EXTRA_TORCH_TENSOR_OPERATION_FIELDS = list(EXTRA_TORCH_TENSOR_OPERATION_MAP.keys())

def convert_module(model, rule='alpha1beta0'):

    for attribute_name in get_nested_nn_modules(model):
        attribute_value = getattr(model, attribute_name)
        setattr(model, attribute_name, lrp_modification_module(attribute_value, rule=rule))

    for extra_field in EXTRA_TORCH_TENSOR_OPERATION_FIELDS:
        if hasattr(model, extra_field):
            print("convert extra_field",extra_field)
            setattr(model, extra_field, EXTRA_TORCH_TENSOR_OPERATION_MAP[extra_field])

    return


def register_lrp_hooks(model):
    """
    registers backward hooks for Transformer2DModel
    to save layer relevances

    must to be called before any forward or backward pass of transformer model

    Saves sequences relevances to attributes:
    ```
        model.source_sequence_relevances            = List[ torch.Tensor ] # mean relevances

        # not works
        model.condition_sequence_relevances         = List[ torch.Tensor ]
        # todo
        model.timesteps_relevances                  = List[ torch.Tensor ]
    ```
    """

    model.source_sequence_relevances            = []
    model.condition_sequence_relevances         = []

    def source_embeddings_hook(layer_name):
        def layers_backward_hook(module, grad_output):
            grad_output = grad_output[0]
            print("source_embeddings_hook hook", layer_name, module)
            print("source_embeddings_hook output", grad_output.shape)

            model.source_sequence_relevances.append(grad_output.detach().cpu())

            return None

        return layers_backward_hook

    model.latent_image_embedding.register_full_backward_pre_hook(source_embeddings_hook('latent_image_embedding'))

    torch.nn.modules.module.register_module_full_backward_hook(lambda module, grad_input, grad_output: print("compute backward for", module, [ gi.sum() for gi in grad_input if gi is not None ], [go.sum() for go in grad_output if go is not None]))

    # def condition_sequence_hook(layer_name):
    #     def layers_backward_hook(module, grad_input, grad_output):
    #         # print("grad_input", grad_input)
    #         encoder_hidden_states_grad = grad_input[0] # encoder_hidden_states
    #         # timesteps tensor should be computed for float tensor
    #         # timesteps_grad = grad_input[1] # timesteps
    #         print("condition_sequence_hook hook", layer_name, module)
    #         model.condition_sequence_relevances.append(encoder_hidden_states_grad.detach().cpu())
    #         # model.timesteps_relevances.append(timesteps_grad.detach().cpu())
    #         return None
    #     return layers_backward_hook

    # model.register_full_backward_hook(condition_sequence_hook('transformer'))

    return

