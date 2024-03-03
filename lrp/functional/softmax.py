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
    input_orig = input.clone()
    relevance_output_orig = relevance_output.clone()
    print("softmax input.shape", input.shape, "dim", ctx.dim)

    batch_size = input.shape[0]
    relevance_output_sum = relevance_output.sum()
    print("softmax relevance_output sum ", relevance_output_sum)

    assert relevance_output_sum < 1.1
    assert relevance_output_sum > 0.0

    original_input_shape = input.shape

    input_len_shape = len(original_input_shape)
    relevance_scaler = input.shape[0]
    if input_len_shape > 2:
        input = input.flatten(0, input_len_shape - 2)
        relevance_scaler = input.shape[0]
        relevance_output = relevance_output.flatten(0, input_len_shape - 2)

    in_features = input.shape[ctx.dim] # number of input features
    batch_size = input.shape[0]

    # print("input", input.shape)

    # z_{ij} = 1/n * f( 0 ) +  df/dx(input)  * input

    zeros = torch.zeros_like(input) + 1e-6

    softmax_module = nn.Softmax(dim=-1)
    # softmax_result = F.softmax(input=input, dim=ctx.dim) # [ bs, in_features ]
    # softmax_jacobian = torch.diag_embed(softmax_result) - torch.outer(softmax_result, softmax_result)

    softmax_output = softmax_module(input) # [ bs ]
    # assert softmax_output.shape == torch.Size([batch_size, in_features])

    # [ bs, 1, in_features ]
    softmax_output_unsqueezed = softmax_output.unsqueeze(1)
    # [ bs, in_features, 1 ]
    softmax_output_unsqueezed_t = softmax_output_unsqueezed.permute(0, 2, 1)

    # bs, in_features, in_features
    softmax_jacobian = torch.bmm(softmax_output_unsqueezed_t, softmax_output_unsqueezed)
    # print("softmax_jacobian min", softmax_jacobian.min(), "max", softmax_jacobian.max()) # [ bs, in_features, in_features ]

    # print("softmax_output_unsqueezed.repeat(1, in_features, 1) * torch.eye(in_features).unsqueeze(batch_size, 1, 1)", softmax_output_unsqueezed.repeat(1, in_features, 1) * torch.eye(in_features).repeat(batch_size, 1, 1))

    #                  - S_i * S_j        + S_i ( if i == j )
    softmax_jacobian = - softmax_jacobian + (softmax_output_unsqueezed.repeat(1, in_features, 1) * torch.eye(in_features).repeat(batch_size, 1, 1))

    # print("softmax_jacobian", "min", softmax_jacobian.min(), "max", softmax_jacobian.max()) # [ bs, in_features, in_features ]
    # assert softmax_jacobian.shape == torch.Size([batch_size, in_features, in_features]), 'softmax_jacobian shape is ok'

    # print("softmax_jacobian", softmax_jacobian)
    # print("input", input)
    z_ij = 1/in_features * F.softmax(zeros+1e-5, dim=ctx.dim) + torch.bmm(softmax_jacobian, input.unsqueeze(2)).squeeze(-1)
    # print("z_ij", z_ij.shape)

    # [ bs, out_features, in_features ]
    total_relevance = alpha_beta_on_z_ij(alpha, beta, z_ij)

    assert total_relevance.sum().item() > 0

    # print("total_relevance", total_relevance.shape)
    # print("softmax total_relevance.sum()", total_relevance.sum())

    relevance_input = relevance_output * total_relevance
    relevance_input = (relevance_input + 1e-6) / ((relevance_input+1e-6).sum(dim=-1, keepdim=True))

    assert relevance_input.shape == input.shape, f"{relevance_input.shape} == {input.shape}"

    trace.do_trace(relevance_input)

    if input_len_shape > 2:
        relevance_input = relevance_input.reshape(original_input_shape)

    print("softmax relevance_scaler", relevance_scaler, "relevance_input", relevance_input.shape)
    relevance_input = relevance_input / relevance_scaler * relevance_output_sum
    print("softmax relevance input sum", relevance_input.sum())

    assert torch.allclose(relevance_output.sum(), relevance_input.sum(), atol=1e-3)

    return relevance_input, None


class SoftmaxAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, input, dim=None):
        return _forward_alpha_beta(ctx, input, dim=dim)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta(1.0, 0.0, ctx, relevance_output)

softmax = {
        "gradient":             F.softmax,
        "epsilon":              NotImplementedError,
        "gamma":                NotImplementedError,
        "gamma+epsilon":        NotImplementedError,
        "alpha1beta0":          SoftmaxAlpha1Beta0.apply,
        "alpha2beta1":          NotImplementedError,
        "patternattribution":   NotImplementedError,
        "patternnet":           NotImplementedError,
}
