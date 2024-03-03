import torch
from .functional import mul

class Mul(torch.nn.Softmax):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rule = "gradient"

    def forward(self, input1, input2, rule="alpha1beta0", **kwargs):
        return mul[self.rule](input1, input2)

    @classmethod
    def from_torch(cls, *args):
        module = cls()
        return module
