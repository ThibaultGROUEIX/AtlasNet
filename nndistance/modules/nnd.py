from torch.nn.modules.module import Module
from functions.nnd import NNDFunction

class NNDModule(Module):
    def forward(self, input1, input2):
        return NNDFunction()(input1, input2)
