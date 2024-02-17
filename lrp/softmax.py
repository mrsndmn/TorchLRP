import torch
from .functional import softmax

class Softmax(torch.nn.Softmax):
    def forward(self, input, rule="alpha1beta0", **kwargs):
        return softmax[rule](input, self.dim)

    @classmethod
    def from_torch(cls, nn_softmax):
        module = cls(dim=nn_softmax.dim)

        return module
