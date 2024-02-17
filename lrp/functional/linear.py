import torch
import torch.nn.functional as F
from torch.autograd import Function

from .utils import identity_fn, gamma_fn, add_epsilon_fn, normalize, alpha_beta_on_z_ij
from .. import trace

def _forward_rho(rho, incr, ctx, input, weight, bias):
    ctx.save_for_backward(input, weight, bias)
    ctx.rho = rho
    ctx.incr = incr
    return F.linear(input, weight, bias)

def _backward_rho(ctx, relevance_output):
    input, weight, bias = ctx.saved_tensors
    rho                 = ctx.rho
    incr                = ctx.incr

    weight, bias     = rho(weight, bias)
    Z                = incr(F.linear(input, weight, bias))

    relevance_output = relevance_output / Z
    relevance_input  = F.linear(relevance_output, weight.t(), bias=None)
    relevance_input  = relevance_input * input

    trace.do_trace(relevance_input) 
    return relevance_input, None, None

class LinearEpsilon(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        return _forward_rho(identity_fn, add_epsilon_fn(0.1), ctx, input, weight, bias) # TODO make batter way of choosing epsilon

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_rho(ctx, relevance_output)

class LinearGamma(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        return _forward_rho(gamma_fn(0.1), add_epsilon_fn(1e-10), ctx, input, weight, bias)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_rho(ctx, relevance_output)

class LinearGammaEpsilon(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        return _forward_rho(gamma_fn(0.1), add_epsilon_fn(1e-1), ctx, input, weight, bias)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_rho(ctx, relevance_output)


def _forward_alpha_beta(ctx, input, weight, bias):
    Z = F.linear(input, weight, bias)
    ctx.save_for_backward(input, weight, bias)
    return Z

def _backward_alpha_beta_explicit(alpha, beta, ctx, relevance_output):
    input, weights, bias = ctx.saved_tensors

    # weights          ~ [ out_features, in_features ]
    # bias             ~ [ out_features ]
    # input            ~ [ bs, in_features ]
    # relevance_output ~ [ bs, out_features ]

    print("linear relevance_output", relevance_output.sum())
    print("linear relevance_output value", relevance_output)

    batch_size, in_features = input.shape[0], input.shape[1]
    out_features = weights.shape[0]

    assert relevance_output.shape == torch.Size([batch_size, out_features]), f"relevance_output.shape={relevance_output.shape}"

    input_unsqueezed = input.unsqueeze(-1) # [ *, in, 1 ]
    assert input_unsqueezed.shape[-1] == 1, f'input_unsqueezed.shape {input_unsqueezed.shape}'
    input_unsqueezed_repeated = input_unsqueezed.repeat(1, 1, out_features) # [ bs, input_features, out_features ]

    weights_unsqueezed = weights.unsqueeze(0) # [ 1, out_features, in_features ]
    weights_unsqueezed_permuted = weights_unsqueezed.permute(0, 2, 1) # [ 1, in_features, out_features ]
    z_ij = input_unsqueezed_repeated * weights_unsqueezed_permuted # [ bs, in_features, out_features ]
    z_ij = z_ij.permute(0, 2, 1) # [ bs, out_features, in_features ]

    assert z_ij.shape == torch.Size([batch_size, out_features, in_features]), f"z_ij.shape {z_ij.shape}"

    # [ bs, out_features, in_features ]
    total_relevance = alpha_beta_on_z_ij(alpha, beta, z_ij)

    # relevance_output.unsqueeze(1) ~ [ bs, 1, out_features ]
    relevance_input = torch.bmm(relevance_output.unsqueeze(1), total_relevance) # [ bs, 1, in_features ]

    relevance_input = relevance_input.squeeze(1)

    assert relevance_input.shape == input.shape, f"{relevance_input.shape} == {input.shape}"

    # print("input", "min", input.min(), "max", input.max())
    print("linear relevance_input", relevance_input.sum())

    trace.do_trace(relevance_input)
    return relevance_input, None, None



def _backward_alpha_beta(alpha, beta, ctx, relevance_output):
    """
        Inspired by https://github.com/albermax/innvestigate/blob/1ed38a377262236981090bb0989d2e1a6892a0b1/innvestigate/analyzer/relevance_based/relevance_rule.py#L270
    """
    input, weights, bias = ctx.saved_tensors

    # weights ~ [ in_features, out_features ]
    # bias    ~ [ out_features ]
    # input   ~ [*, input_features]
    # Z       ~ [ *, out_features ]
    print("linear relevance_output", relevance_output.sum())

    sel = weights > 0
    zeros = torch.zeros_like(weights)
    weights_pos       = torch.where(sel,  weights, zeros)
    weights_neg       = torch.where(~sel, weights, zeros)

    input_pos         = torch.where(input >  0, input, torch.zeros_like(input))
    input_neg         = torch.where(input <= 0, input, torch.zeros_like(input))

    def f(X1, X2, W1, W2, bias=None):

        Z1  = F.linear(X1, W1, bias=None)
        Z2  = F.linear(X2, W2, bias=None)
        Z   = Z1 + Z2 # Z_j^{+/-} # [ *, out_features ]

        # По статье войты тут должно быть еще добавление bias
        # Но такое добавление сломает правило нормы и релевантность не будет
        # Константной

        # R_j / Z_j^{+/-}
        rel_out = relevance_output / (Z + (Z==0).float()* 1e-6)

        # Осталось вычислить
        # R_j / Z_j^{+/-} * Z_{ij}
        #
        # Сделаем это последовательно: Z_{ij} = W_{ij}.T @ X_i

        t1 = F.linear(rel_out, W1.t(), bias=None)
        t2 = F.linear(rel_out, W2.t(), bias=None)

        # R_j / Z_j^{+/-}
        r1  = t1 * X1
        r2  = t2 * X2

        return r1 + r2

    # положительная релевантность для множителей с одинаковыми знаками
    pos_rel         = f(input_pos, input_neg, weights_pos, weights_neg)
    # отрицательная релевантность для множителей с разными знаками
    neg_rel         = f(input_neg, input_pos, weights_pos, weights_neg)

    relevance_input = pos_rel * alpha - neg_rel * beta

    trace.do_trace(relevance_input)

    print("linear relevance_input", relevance_input.sum())

    return relevance_input, None, None

class LinearAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        return _forward_alpha_beta(ctx, input, weight, bias)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta(1., 0., ctx, relevance_output)

class LinearAlpha1Beta0Explicit(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        return _forward_alpha_beta(ctx, input, weight, bias)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta_explicit(1., 0., ctx, relevance_output)


class LinearAlpha2Beta1(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        return _forward_alpha_beta(ctx, input, weight, bias)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta(2., 1., ctx, relevance_output)


def _forward_pattern(attribution, ctx, input, weight, bias, pattern):
    ctx.save_for_backward(input, weight, pattern)
    ctx.attribution = attribution
    return F.linear(input, weight, bias)

def _backward_pattern(ctx, relevance_output):
    input, weight, P = ctx.saved_tensors

    if  ctx.attribution: P = P * weight # PatternAttribution
    relevance_input  = F.linear(relevance_output, P.t(), bias=None)

    trace.do_trace(relevance_input)
    return relevance_input, None, None, None

class LinearPatternAttribution(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, pattern=None):
        return _forward_pattern(True, ctx, input, weight, bias, pattern) 

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_pattern(ctx, relevance_output)

class LinearPatternNet(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, pattern=None):
        return _forward_pattern(False, ctx, input, weight, bias, pattern) 

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_pattern(ctx, relevance_output)

linear = {
        "gradient":             F.linear,
        "epsilon":              LinearEpsilon.apply,
        "gamma":                LinearGamma.apply,
        "gamma+epsilon":        LinearGammaEpsilon.apply,
        "alpha1beta0":          LinearAlpha1Beta0Explicit.apply,
        "alpha2beta1":          LinearAlpha2Beta1.apply,
        "patternattribution":   LinearPatternAttribution.apply,
        "patternnet":           LinearPatternNet.apply,
}
