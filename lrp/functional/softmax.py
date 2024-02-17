import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn

from .utils import alpha_beta_on_z_ij
from .. import trace

from torch.autograd.functional import jacobian

def _forward_alpha_beta(ctx, input, dim=None):
    Z = F.softmax(input, dim=dim)

    ctx.dim = dim

    ctx.save_for_backward(input)
    return Z

def _backward_alpha_beta(alpha, beta, ctx, relevance_output):
    # [ bs, in_features ]
    input = ctx.saved_tensors[0]

    # batch_size, in_features = input.shape

    print("softmax relevance_output sum ", relevance_output.sum())

    original_input_shape = input.shape

    input_len_shape = len(original_input_shape)
    relevance_scaler = 1
    if input_len_shape > 2:
        batch_size = input.shape[0]
        input = input.flatten(0, input_len_shape - 2)
        relevance_scaler = input.shape[0] / batch_size
        relevance_output = relevance_output.flatten(0, input_len_shape - 2)

    n = input.shape[ctx.dim] # number of input features
    # print("input", input.shape)

    # z_{ij} = 1/n * f( 0 ) +  df/dx(input)  * input

    zeros = torch.zeros_like(input) + 1e-6

    softmax_module = nn.Softmax(dim=ctx.dim)
    # softmax_result = F.softmax(input=input, dim=ctx.dim) # [ bs, in_features ]
    # softmax_jacobian = torch.diag_embed(softmax_result) - torch.outer(softmax_result, softmax_result)
    input_jacobians = []
    for i in range(input.shape[0]):
        layer_norm_jacobian = jacobian(softmax_module, input[i:i+1, :]).squeeze(2)
        input_jacobians.append(layer_norm_jacobian)
        # print("layer_norm_jacobian", layer_norm_jacobian.shape)

    softmax_jacobian = torch.vstack(input_jacobians)

    # print("softmax_jacobian", softmax_jacobian.shape) # [ bs, in_features, in_features ]
    # assert softmax_jacobian.shape == torch.Size([batch_size, in_features, in_features]), 'softmax_jacobian shape is ok'

    # print("softmax_jacobian", softmax_jacobian)
    # print("input", input)
    z_ij = 1/n * F.softmax(zeros, dim=ctx.dim) + torch.bmm(softmax_jacobian, input.unsqueeze(2)).squeeze(-1)
    # print("z_ij", z_ij.shape)

    # [ bs, out_features, in_features ]
    total_relevance = alpha_beta_on_z_ij(alpha, beta, z_ij)

    # print("total_relevance", total_relevance.shape)
    # print("total_relevance", total_relevance)

    relevance_input = relevance_output * total_relevance
    relevance_input = relevance_input / relevance_input.sum(dim=-1, keepdim=True)

    assert relevance_input.shape == input.shape, f"{relevance_input.shape} == {input.shape}"

    trace.do_trace(relevance_input)

    if input_len_shape > 2:
        relevance_input = relevance_input.reshape(original_input_shape)

    print("relevance_scaler", relevance_scaler, "relevance_input", relevance_input.shape)
    relevance_input = relevance_input / relevance_scaler

    print("softmax relevance input sum", relevance_input.sum())

    return relevance_input, None


class SoftmaxAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, input, dim=None):
        return _forward_alpha_beta(ctx, input, dim=dim)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta(1.0, 0.0, ctx, relevance_output)

softmax = {
        "gradient":             F.linear,
        "epsilon":              NotImplementedError,
        "gamma":                NotImplementedError,
        "gamma+epsilon":        NotImplementedError,
        "alpha1beta0":          SoftmaxAlpha1Beta0.apply,
        "alpha2beta1":          NotImplementedError,
        "patternattribution":   NotImplementedError,
        "patternnet":           NotImplementedError,
}
