import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn

from .utils import alpha_beta_on_z_ij
from .. import trace

from torch.autograd.functional import jacobian

def _forward_alpha_beta(ctx, input1, input2):
    Z = input1 + input2

    ctx.save_for_backward(input1, input2)
    return Z

def _backward_alpha_beta(ctx, relevance_output):
    # [ bs, in_features ]
    input1, input2 = ctx.saved_tensors

    relevance_input = relevance_output / 2

    trace.do_trace(relevance_input)

    return relevance_input, relevance_input


class ResidualAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        return _forward_alpha_beta(ctx, input1, input2)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta(ctx, relevance_output)

softmax = {
        "gradient":             F.linear,
        "epsilon":              NotImplementedError,
        "gamma":                NotImplementedError,
        "gamma+epsilon":        NotImplementedError,
        "alpha1beta0":          ResidualAlpha1Beta0.apply,
        "alpha2beta1":          NotImplementedError,
        "patternattribution":   NotImplementedError,
        "patternnet":           NotImplementedError,
}
