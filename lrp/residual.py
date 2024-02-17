import torch
from .functional import residual

class Residual(torch.nn.Softmax):
    def forward(self, input1, input2, rule="alpha1beta0", **kwargs):
        return residual[rule](input1, input2)

    @classmethod
    def from_torch(cls, *args):
        module = cls()
        return module
