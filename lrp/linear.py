import torch
from .functional import linear

class Linear(torch.nn.Linear):
    def forward(self, input, *args, **kwargs):
        rule = "alpha1beta0"
        return linear[rule](input, self.weight, self.bias)

    @classmethod
    def from_torch(cls, lin):
        bias = lin.bias is not None
        module = cls(in_features=lin.in_features, out_features=lin.out_features, bias=bias)
        module.load_state_dict(lin.state_dict())

        return module
