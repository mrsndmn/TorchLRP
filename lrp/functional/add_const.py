import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn

from .utils import alpha_beta_on_z_ij
from .. import trace

def _forward_alpha_beta(ctx, input1, const):
    Z = input1.clone() + const

    # ctx.save_for_backward(input1.clone())
    return Z

def _backward_alpha_beta(ctx, relevance_output):
    # [ bs, in_features ]
    # input1 = ctx.saved_tensors

    return relevance_output.clone(), None


class AddConstAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        return _forward_alpha_beta(ctx, input1, input2)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta(ctx, relevance_output)

add_const = {
    "gradient":          lambda x, y: x + y,
    "alpha1beta0":          AddConstAlpha1Beta0.apply,
}
