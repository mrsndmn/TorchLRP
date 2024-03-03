import torch
from .functional.layer_norm import layer_norm

class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rule = "gradient"

    def forward(self, input, **kwargs):
        return layer_norm[self.rule](input, self.normalized_shape, self.weight, self.bias)

    @classmethod
    def from_torch(cls, layer_norm):
        module = cls(
            normalized_shape=layer_norm.normalized_shape,
            eps=layer_norm.eps,
            elementwise_affine=layer_norm.elementwise_affine
        )

        return module
