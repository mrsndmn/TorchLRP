import torch
from .functional import linear

class Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rule = "gradient"

    def forward(self, input, *args, **kwargs):
        return linear[self.rule](input, self.weight, self.bias)

    @classmethod
    def from_torch(cls, lin, rule):
        bias = lin.bias is not None
        module = cls(in_features=lin.in_features, out_features=lin.out_features, bias=bias)
        module.load_state_dict(lin.state_dict())
        module.rule = rule

        return module
