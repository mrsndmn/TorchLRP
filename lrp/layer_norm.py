import torch
from .functional.layer_norm import layer_norm

class LayerNorm(torch.nn.LayerNorm):
    def forward(self, input, **kwargs):
        rule = "alpha1beta0"
        return layer_norm[rule](input, self.normalized_shape, self.weight, self.bias)

    @classmethod
    def from_torch(cls, layer_norm):
        module = cls(
            normalized_shape=layer_norm.normalized_shape,
            eps=layer_norm.eps,
            elementwise_affine=layer_norm.elementwise_affine
        )

        return module
