import torch
from torch.autograd import Function
import torch.nn as nn
import torch.autograd as Variable
import torchsnooper


class transform(nn.Module):
    def __init__(self, num, cls):
        super(transform, self).__init__()
        self.batch_size = num
        self.cls = cls

    # @torchsnooper.snoop()
    def forward(self, input):
        _, c, w, h = input.shape
        return input.view(self.batch_size, self.cls, -1, c, w, h)