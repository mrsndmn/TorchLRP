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
    if isinstance(normalized_shape, list):
        assert len(normalized_shape) == 1, f"len(normalized_shape)={len(normalized_shape)}"

    ctx.save_for_backward(input, weight, bias)
    return Z

def _backward_alpha_beta(alpha, beta, ctx, relevance_output):
    # [ bs, in_features ]
    input, weight, bias = ctx.saved_tensors

    original_input_shape = input.shape

    input_len_shape = len(original_input_shape)
    relevance_scaler = 1
    if input_len_shape > 2:
        batch_size = input.shape[0]
        input = input.flatten(0, input_len_shape - 2)
        relevance_scaler = input.shape[0]
        relevance_output = relevance_output.flatten(0, input_len_shape - 2)

    # print("input", input.shape)
    relevance_output_sum = relevance_output.sum()
    print("layer norm input", input.shape)
    print("layer norm relevance_output", relevance_output_sum)

    input_2_dim_shape = input.shape

    normalized_shape = ctx.normalized_shape

    # z_{ij} = 1/n * f( 0 ) +  df/dx(input)  * input
    eps = 1e-5

    zeros = torch.zeros(input_2_dim_shape) + eps

    layer_norm_module = nn.LayerNorm(normalized_shape)
    if weight is not None:
        layer_norm_module.weight = nn.Parameter(weight)
    if bias is not None:
        layer_norm_module.bias =   nn.Parameter(bias)


    mean_variance_dims = [ -1 ]
    n_for_mean_variance = normalized_shape
    if isinstance(normalized_shape, (list, tuple)):
        mean_variance_dims = list(range(-len(normalized_shape), 0))
        n_for_mean_variance = 1
        for dim in normalized_shape:
            n_for_mean_variance = dim * n_for_mean_variance

    print("layer norm: mean_variance_dims=", mean_variance_dims, "n_for_mean_variance=", n_for_mean_variance)

    batch_variance = torch.var(input, dim=mean_variance_dims, keepdim=True, unbiased=False)
    batch_mean     = torch.mean(input, dim=mean_variance_dims, keepdim=True)

    # [ bs, embedding_dim, embedding_dim ]
    eye_mask = torch.eye(n_for_mean_variance).unsqueeze(0).repeat(input.shape[0], 1, 1)

    # [ bs, embedding_dim ]
    input_jacobians_numerator_eye = ((n_for_mean_variance - 1) / n_for_mean_variance * batch_variance) - (input - batch_mean) / (n_for_mean_variance - 1)
    input_jacobians_numerator_eye = input_jacobians_numerator_eye.unsqueeze(2).repeat(1, 1, n_for_mean_variance)
    # [ bs, embedding_dim, embedding_dim ]
    input_jacobians_numerator_eye[ eye_mask == 0 ] = 0

    # [ bs, embedding_dim ]
    input_jacobians_numerator_out_of_eye = (-1 / n_for_mean_variance * batch_variance) - (input - batch_mean) / (n_for_mean_variance - 1)
    # [ bs, embedding_dim, embedding_dim ]
    input_jacobians_numerator_out_of_eye = input_jacobians_numerator_out_of_eye.unsqueeze(2).repeat(1, 1, n_for_mean_variance)

    # combine eye and out of eye
    input_jacobians_numerator = input_jacobians_numerator_out_of_eye
    input_jacobians_numerator[ eye_mask.bool() ] = input_jacobians_numerator_eye[ eye_mask.bool() ]

    # d LayerNorm / d x_i
    # [ bs, embedding_dim, embedding_dim ]
    # print("batch_variance", batch_variance.shape)
    # print("input_jacobians_numerator", input_jacobians_numerator.shape)
    batch_variance_unsqueezed = batch_variance.unsqueeze(2)
    input_jacobians = input_jacobians_numerator / torch.sqrt( batch_variance_unsqueezed + eps ) * (batch_variance_unsqueezed + eps)

    if weight is not None:
        input_jacobians *= layer_norm_module.weight.unsqueeze(0)

    # [ bs, embedding_dim, 1 ]
    z_ij = 1/normalized_shape[0] * layer_norm_module(zeros) + torch.bmm(input_jacobians, input.unsqueeze(2)).squeeze(-1)
    # print("z_ij", z_ij.shape)

    total_relevance = alpha_beta_on_z_ij(alpha, beta, z_ij)

    # print("total_relevance", total_relevance.shape)

    relevance_input = relevance_output * total_relevance + 1e-6
    relevance_input = relevance_input / (relevance_input.sum(dim=-1, keepdim=True))

    assert relevance_input.shape == input.shape, f"{relevance_input.shape} == {input.shape}"

    trace.do_trace(relevance_input)

    if input_len_shape > 2:
        relevance_input = relevance_input.reshape(original_input_shape)

    print("layer norm relevance_scaler", relevance_scaler, "relevance_input", relevance_input.shape)
    relevance_input = relevance_input / relevance_scaler * relevance_output_sum

    print("layer norm relevance_input", relevance_input.sum())

    assert torch.allclose(relevance_output.sum(), relevance_input.sum(), atol=1e-3)

    return relevance_input, None, None, None


class LayerNormAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias):
        return _forward_alpha_beta(ctx, input, normalized_shape, weight, bias)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta(1.0, 0.0, ctx, relevance_output)

layer_norm = {
        "gradient":             F.layer_norm,
        "epsilon":              NotImplementedError,
        "gamma":                NotImplementedError,
        "gamma+epsilon":        NotImplementedError,
        "alpha1beta0":          LayerNormAlpha1Beta0.apply,
        "alpha2beta1":          NotImplementedError,
        "patternattribution":   NotImplementedError,
        "patternnet":           NotImplementedError,
}
