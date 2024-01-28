import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn

from .utils import alpha_beta_on_z_ij
from .. import trace

from torch.autograd.functional import jacobian

def _forward_alpha_beta(ctx, input, normalized_shape, weight, bias):
    Z = F.layer_norm(input, normalized_shape, weight, bias)

    ctx.normalized_shape = normalized_shape

    ctx.save_for_backward(input, weight, bias)
    return Z

def _backward_alpha_beta(alpha, beta, ctx, relevance_output):
    # [ bs, in_features ]
    input, weight, bias = ctx.saved_tensors

    normalized_shape = ctx.normalized_shape

    # z_{ij} = 1/n * f( 0 ) +  df/dx(input)  * input

    zeros = torch.zeros_like(input) + 1e-6

    layer_norm_module = nn.LayerNorm(normalized_shape)
    layer_norm_module.weight = nn.Parameter(weight)
    layer_norm_module.bias =   nn.Parameter(bias)


    # softmax_result = F.softmax(input=input, dim=ctx.dim) # [ bs, in_features ]
    # layer_norm_jacobian = torch.diag_embed(softmax_result) - torch.outer(softmax_result, softmax_result)
    layer_norm_jacobian = jacobian(layer_norm_module, input).squeeze(2)
    # print("layer_norm_jacobian", layer_norm_jacobian.shape) # [ bs, in_features, in_features ]

    # print("layer_norm_jacobian", layer_norm_jacobian)
    # print("input", input)
    z_ij = 1/normalized_shape[0] * layer_norm_module(zeros) + torch.bmm(layer_norm_jacobian, input.unsqueeze(2)).squeeze(-1)
    # print("z_ij", z_ij.shape)

    # [ bs, out_features, in_features ]
    total_relevance = alpha_beta_on_z_ij(alpha, beta, z_ij)

    # print("total_relevance", total_relevance.shape)
    # print("total_relevance", total_relevance)
    # print("relevance_output", relevance_output.shape)
    # print("relevance_output", relevance_output)

    relevance_input = relevance_output * total_relevance
    relevance_input = relevance_input / relevance_input.sum(dim=-1)

    assert relevance_input.shape == input.shape, f"{relevance_input.shape} == {input.shape}"

    trace.do_trace(relevance_input)

    return total_relevance, None, None, None


class LayerNormAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias):
        return _forward_alpha_beta(ctx, input, normalized_shape, weight, bias)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta(1.0, 0.0, ctx, relevance_output)

softmax = {
        "gradient":             F.linear,
        "epsilon":              NotImplementedError,
        "gamma":                NotImplementedError,
        "gamma+epsilon":        NotImplementedError,
        "alpha1beta0":          LayerNormAlpha1Beta0.apply,
        "alpha2beta1":          NotImplementedError,
        "patternattribution":   NotImplementedError,
        "patternnet":           NotImplementedError,
}
