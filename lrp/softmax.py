import torch
from .functional import softmax

class Softmax(torch.nn.Softmax):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rule = "gradient"

    def forward(self, input, **kwargs):
        return softmax[self.rule](input, self.dim)

    @classmethod
    def from_torch(cls, nn_softmax, rule):
        module = cls(dim=nn_softmax.dim)
        module.rule = rule

        return module
