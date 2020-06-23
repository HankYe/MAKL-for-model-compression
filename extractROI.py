import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import torchsnooper
from torch.autograd import Variable
from roi_pooling.functions.roi_pooling import roi_pooling_2d


class roi_extract(nn.Module):
    def __init__(self, Channel, device=0):
        super(roi_extract, self).__init__()
        self.Channel = Channel
        self.device = device

    # @torchsnooper.snoop()
    def forward(self, T_feature, S_feature, input):
        """
           ROI Extraction layer
        """
        assert T_feature.shape[-1] == S_feature.shape[-1]
        n, cls, top_k, loc = input.shape
        decode_box = input.view(n, -1, loc)[:, :, 1:]
        # score = input.view(n, -1, loc)[:, :, 0]
        total_rois = torch.Tensor([]).to(self.device)
        for i in range(n):
            img_index = torch.ones(decode_box.shape[1], 1) * i
            rois = torch.cat((img_index.to(self.device), decode_box[i].to(self.device)), dim=1)
            total_rois = torch.cat((total_rois, rois), dim=0)

        n, _, w, h = T_feature.size() # T_feature map of 38*38
        T_feat = T_feature.view(n, -1, w * h).permute(0, 2, 1)
        S_feat = S_feature.view(n, -1, w * h).permute(0, 2, 1)
        T_pooled = F.adaptive_max_pool1d(T_feat, self.Channel)
        S_pooled = F.adaptive_max_pool1d(S_feat, self.Channel)
        _, _, c = T_pooled.size()
        T_pooled = T_pooled.permute(0, 2, 1)
        S_pooled = S_pooled.permute(0, 2, 1)
        out_featT = T_pooled.view(n, c, w, h)
        out_featS = S_pooled.view(n, c, w, h)

        return out_featT, out_featS, total_rois
        # out7 = roi_pooling_2d(out_feat, Variable(total_rois, requires_grad=False), output_size=(7, 7), spatial_scale=1 / 8)
        # _, c, w, h = out7.shape
        # input_dis = out7.view(n, cls, -1, c, w, h)
        #
        # return input_dis

class roi_extract_multiscale(nn.Module):
    def __init__(self, Channel, device=0):
        super(roi_extract_multiscale, self).__init__()
        self.Channel = Channel
        self.device = device

    # @torchsnooper.snoop()
    def forward(self, T_feature, S_feature, input):
        """
           ROI Extraction layer
        """
        n, cls, top_k, loc = input.shape
        decode_box = input.view(n, -1, loc)[:, :, 1:]
        # score = input.view(n, -1, loc)[:, :, 0]
        total_rois = torch.Tensor([]).to(self.device)
        for i in range(n):
            img_index = torch.ones(decode_box.shape[1], 1) * i
            rois = torch.cat((img_index.to(self.device), decode_box[i].to(self.device)), dim=1)
            total_rois = torch.cat((total_rois, rois), dim=0)

        out_featT = []
        out_featS = []
        for T_feat,  S_feat in zip(T_feature, S_feature):
            n, _, w, h = T_feat.size()
            T_feat = T_feat.view(n, -1, w * h).permute(0, 2, 1)
            S_feat = S_feat.view(n, -1, w * h).permute(0, 2, 1)
            T_pooled = F.adaptive_max_pool1d(T_feat, self.Channel)
            S_pooled = F.adaptive_max_pool1d(S_feat, self.Channel)
            _, _, c = T_pooled.size()
            T_pooled = T_pooled.permute(0, 2, 1).view(n, c, w, h)
            S_pooled = S_pooled.permute(0, 2, 1).view(n, c, w, h)
            out_featT.append(Variable(T_pooled))
            out_featS.append(Variable(S_pooled, requires_grad=True))

        return out_featT, out_featS, total_rois