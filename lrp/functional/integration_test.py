from copy import deepcopy
import pytest
import torch
import torch.nn as nn

from lrp.functional.integration import convert_module, register_lrp_hooks

from lrp.residual import Residual as LRPResidual

class IntegrationModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = nn.ReLU()
        self.lin1 = nn.Linear(20, 40)
        self.lin2 = nn.Linear(40, 40)
        self.out = nn.Linear(40, 10)

        self.softmax = nn.Softmax(dim=-1)

        self.resudual = LRPResidual()

        return

    def forward(self, input):

        latent1 = self.activation(self.lin1(input))
        latent2 = self.activation(self.lin2(latent1))
        output = self.out(self.resudual(latent1, latent2))

        return self.softmax(output)

def test_lrp_attention():

    from diffusers.models.attention_processor import Attention, AttnProcessor2_0

    inner_dim = 30
    attention = Attention(
        query_dim=inner_dim,
    )
    attention.eval()
    attention_original = deepcopy(attention)

    print("attention", attention)

    convert_module(attention)

    hidden_states = torch.rand([10, 7, inner_dim], requires_grad=True) # [bs, seq_len, hidden_dim]
    attention_out = attention.forward(hidden_states)
    attention_original_out = attention_original.forward(hidden_states)

    assert torch.allclose(attention_out, attention_original_out, atol=1e-6), 'blocks outputs are the same'

    print("attention_out", attention_out.shape)

    attention_out.backward( torch.ones_like(attention_out) / attention_out.numel() )

    hidden_states_grad_sum = hidden_states.grad.sum()
    print("hidden_states_grad_sum", hidden_states_grad_sum)

    # assert torch.allclose(hidden_states_grad_sum, torch.tensor(1.), atol=1e-4), 'hidden_states_grad_sum is 1'

    return



def test_lrp_transformer_block():

    from diffusers.models.attention import BasicTransformerBlock

    inner_dim = 30
    transformer_block = BasicTransformerBlock(
        dim=inner_dim,
        num_attention_heads=3,
        attention_head_dim=10,
    )
    transformer_block.eval()
    transformer_block_original = deepcopy(transformer_block)

    print("transformer_block", transformer_block)

    convert_module(transformer_block)

    hidden_states = torch.rand([1, 3, inner_dim], requires_grad=True) # [bs, seq_len, hidden_dim]
    transformer_block_out = transformer_block.forward(hidden_states)
    transformer_block_original_out = transformer_block_original.forward(hidden_states)

    assert torch.allclose(transformer_block_out, transformer_block_original_out, atol=1e-6), 'blocks outputs are the same'

    print("transformer_block_out", transformer_block_out.shape)

    transformer_block_out.backward( torch.ones_like(transformer_block_out) / transformer_block_out.numel() )

    hidden_states_grad_sum = hidden_states.grad.sum()
    print("hidden_states_grad_sum", hidden_states_grad_sum)

    # assert hidden_states_grad_sum == 1, 'hidden_states_grad_sum is 1'

    return

def test_lrp_transformer_model():

    from diffusers import Transformer2DModel

    cross_attention_dim = 128
    model_kwargs = {
        "attention_bias": True,
        "cross_attention_dim": cross_attention_dim,
        "attention_head_dim": 64,
        "num_attention_heads": 2,
        "num_vector_embeds": 10,
        "num_embeds_ada_norm": 100,
        "sample_size": 32,
        "height": 32,
        "num_layers": 2,
        "activation_fn": "geglu-approximate",
        "output_attentions": True,
        "dropout": 0,
    }
    model = Transformer2DModel(**model_kwargs)

    model.train()
    model_original = deepcopy(model)

    convert_module(model)

    batch_size = 3
    image_size = 32
    hidden_states = torch.randint(0, 10, [batch_size, image_size]) # [bs, seq_len, hidden_dim]
    encoder_hidden_states = torch.rand([batch_size, 3, cross_attention_dim], requires_grad=True) # [bs, seq_len, hidden_dim]
    timesteps = torch.randint(0, 100, [batch_size])

    register_lrp_hooks(model)

    transformer_out = model.forward(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timesteps
    )
    transformer_original_out = model_original.forward(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timesteps,
    )

    transformer_out_sample = transformer_out.sample
    print("transformer_out", transformer_out_sample.shape)

    assert torch.allclose(transformer_out_sample, transformer_original_out.sample, atol=1e-6), 'blocks outputs are the same'

    transformer_out_sample.backward( torch.ones_like(transformer_out_sample) / transformer_out_sample.numel() )

    print("model.source_sequence_relevances", model.source_sequence_relevances[0].sum())
    # print("model.condition_sequence_relevances", model.condition_sequence_relevances[0].sum())
    print("encoder_hidden_states", encoder_hidden_states.grad.sum())

    assert len(model.source_sequence_relevances) == 1
    # assert len(model.condition_sequence_relevances) == 1

    assert model.source_sequence_relevances[0].shape == torch.Size([batch_size, image_size, cross_attention_dim])
    # assert model.condition_sequence_relevances[0].shape == encoder_hidden_states.shape
    assert encoder_hidden_states.grad.shape == encoder_hidden_states.shape

    return


def test_lrp_mlp():
    m = IntegrationModule()
    convert_module(m)

    bs = 1
    in_features = m.lin1.in_features
    input = torch.rand([bs, in_features], requires_grad=True)

    print("input sum", input.sum(), "mean", input.mean(), "min", input.min(), "max", input.max())

    latent1 = m.activation(m.lin1(input))
    latent2 = m.activation(m.lin2(latent1))
    output = m.out(m.resudual(latent1, latent2))
    probas = m.softmax(output)

    probas.backward(torch.ones_like(probas))

    input_grad_sum = input.grad.sum(dim=-1)
    print(input_grad_sum)

    assert torch.allclose(input_grad_sum, torch.ones_like(input_grad_sum), atol=1e-3), 'relevance amount did not changed'

