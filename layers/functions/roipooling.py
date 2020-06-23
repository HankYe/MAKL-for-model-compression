"""
Author: Hancheng Ye
this file processes the ROI pooling and extraction of ROI
"""
import torch
import torch.tensor as Tensor
import torch.nn as nn
from torch.autograd import Function

class ROI_PROCESS(nn.Module):
    def __init__(self, scale, output_size=7):
        super(ROI_PROCESS, self).__init__()
        self.scale = scale
        self.roipooling = nn.AdaptiveMaxPool2d(output_size=output_size, return_indices=True)
        self.extractROI = extractROI(self.scale)

    def forward(self, input,):
        x = self.extractROI()




class extractROI(Function):
    def __init__(self, scale = 1):
        super(extractROI, self).__init__()
        self.scale = scale

    def forward(ctx, x, rois):
        rois = rois * ctx.scale

