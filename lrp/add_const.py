import torch
import torch.nn as nn
from .functional import add_const

class AddConst(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rule = "gradient"

    def forward(self, input1, const, **kwargs):
        return add_const[self.rule](input1, const)

    @classmethod
    def from_torch(cls, *args):
        module = cls()
        return module
