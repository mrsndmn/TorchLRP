import torch
import torch.nn.functional as F
from torch.autograd import Function

from .utils import alpha_beta_on_z_ij
from .. import trace

def _forward_alpha_beta(ctx, input1, scale):
    Z = input1 * scale
    return Z

def _backward_alpha_beta_explicit(alpha, beta, ctx, relevance_output):

    return relevance_output, None


class ScalerAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        return _forward_alpha_beta(ctx, input1, input2)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta_explicit(1., 0., ctx, relevance_output)

matmul = {
        "alpha1beta0":          ScalerAlpha1Beta0.apply,
}
