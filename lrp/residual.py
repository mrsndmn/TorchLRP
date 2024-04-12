import torch
import torch.nn as nn
from .functional import residual

class Residual(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rule = "gradient"

    def forward(self, input1, input2, **kwargs):
        return residual[self.rule](input1, input2)

    @classmethod
    def from_torch(cls, *args):
        module = cls()
        return module
