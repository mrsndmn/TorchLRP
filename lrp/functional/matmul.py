import torch
import torch.nn.functional as F
from torch.autograd import Function

from .utils import alpha_beta_on_z_ij
from .. import trace

def _forward_alpha_beta(ctx, input1, input2):
    Z = input1 @ input2
    ctx.save_for_backward(input1, input2)
    return Z

def _backward_alpha_beta_explicit(alpha, beta, ctx, relevance_output):
    input1, input2 = ctx.saved_tensors

    # input1           ~ [ *, in_features, hidden_dim ]
    # input2           ~ [ *, hidden_dim, out_features ]
    # relevance_output ~ [ *, in_features, out_features ]

    assert len(input1.shape) == len(input2.shape), 'inputs has the same dims count'

    relevance_output_sum = relevance_output.sum()
    print("matmul relevance_output", relevance_output.sum())
    print("matmul input1", input1.shape)
    print("matmul input2", input2.shape)

    # print("linear relevance_output shape", relevance_output.shape)

    original_input1_shape = input1.shape # # [ bs, in_features, hidden ]
    original_input2_shape = input2.shape # # [ bs, hidden, out_features ]
    original_relevance_output_shape = relevance_output.shape # [ bs, in_features, out_features ]

    input1_len_shape = len(original_input1_shape)
    if input1_len_shape > 2:
        input1 = input1.flatten(0, input1_len_shape - 3)

    input2_len_shape = len(original_input2_shape)
    if input2_len_shape > 2:
        input2 = input2.flatten(0, input2_len_shape - 3)

    out_relevance_len_shape = len(original_relevance_output_shape)
    if out_relevance_len_shape > 2:
        relevance_output = relevance_output.flatten(0, out_relevance_len_shape - 3)

    # print('input1.shape', input1.shape)
    # print('input2.shape', input2.shape)
    # print('relevance_output', relevance_output.shape)
    batch_size, in_features, hidden_dim = input1.shape[0], input1.shape[1], input1.shape[2]
    out_features = input2.shape[2]

    assert relevance_output.shape == torch.Size([batch_size, in_features, out_features]), f"relevance_output.shape={relevance_output.shape}"

    input1_unsqueezed = input1.unsqueeze(-1) # [ bs, in_features, hidden, 1 ]

    input1_unsqueezed_repeated = input1_unsqueezed.repeat(1, 1, 1, out_features) # [ bs, input_features, hidden, out_features ]

    # input2 [ bs, 1, hidden, out_features ]
    input2_unsqeezed_repeated = input2.unsqueeze(1).repeat(1, in_features, 1, 1)
    z_ij = input1_unsqueezed_repeated * input2_unsqeezed_repeated # [ bs, in_features, hidden, out_features ]
    z_ij = z_ij.permute(0, 1, 3, 2) # [ bs, in_features, out_features, hidden ]

    expected_z_ij_shape = torch.Size([batch_size, in_features, out_features, hidden_dim])
    assert z_ij.shape == expected_z_ij_shape, f"z_ij.shape={z_ij.shape}, expected shape={expected_z_ij_shape}"

    # [ bs, in_features, out_features, hidden ] # normed by hidden
    total_relevance = alpha_beta_on_z_ij(alpha, beta, z_ij)
    # print("relevance_output.shape", relevance_output.shape)

    # [ bs, in_features, 1, out_features ] @ [ bs, in_features, out_features, hidden ]
    relevance_input1 = torch.bmm(relevance_output.unsqueeze(2).flatten(0, 1), total_relevance.flatten(0, 1)) # [ bs, in_features, 1, hidden_dim ]
    relevance_input1 = relevance_input1.reshape(batch_size, in_features, 1, hidden_dim)
    relevance_input1 = relevance_input1.squeeze(2)

    # [ bs, out_features, 1, in_features ] @ [ bs, out_features, in_features, hidden ]
    relevance_input2 = torch.bmm(relevance_output.permute(0, 2, 1).unsqueeze(2).flatten(0, 1), total_relevance.permute(0, 2, 1, 3).flatten(0, 1)) # [ bs, out_features, 1, hidden_dim ]
    relevance_input2 = relevance_input2.reshape(batch_size, out_features, 1, hidden_dim)
    relevance_input2 = relevance_input2.squeeze(2) # [ bs, out_features, hidden_dim ]
    relevance_input2 = relevance_input2.permute(0, 2, 1) # [ bs, hidden_dim, out_features ]

    assert relevance_input1.shape == input1.shape, f"{relevance_input1.shape} == {input1.shape}"
    assert relevance_input2.shape == input2.shape, f"{relevance_input2.shape} == {input2.shape}"

    # print("input", "min", input.min(), "max", input.max())

    if input1_len_shape > 2:
        relevance_input1 = relevance_input1.reshape(original_input1_shape)

    if input2_len_shape > 2:
        relevance_input2 = relevance_input2.reshape(original_input2_shape)


    # trace.do_trace(relevance_input1)
    # trace.do_trace(relevance_input2)
    relevance_input1 = relevance_input1 / relevance_input1.sum() / 2 * relevance_output_sum
    relevance_input2 = relevance_input2 / relevance_input2.sum() / 2 * relevance_output_sum

    print("matmul relevance_input1", relevance_input1.sum())
    print("matmul relevance_input2", relevance_input2.sum())

    return relevance_input1, relevance_input2


class MatMulAlpha1Beta0(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        return _forward_alpha_beta(ctx, input1, input2)

    @staticmethod
    def backward(ctx, relevance_output):
        return _backward_alpha_beta_explicit(1., 0., ctx, relevance_output)

matmul = {
        "gradient":             F.linear,
        "alpha1beta0":          MatMulAlpha1Beta0.apply,
}
