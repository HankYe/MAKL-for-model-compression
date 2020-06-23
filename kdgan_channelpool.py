import torch.nn as nn
from torch.autograd import Variable
from data import *
from layers import *
from layers.functions.detect_for_KDGAN import Discriminator
from extractROI import roi_extract_multiscale
from roi_pooling.functions.roi_pooling import roi_pooling_2d


class KDGAN_Feat(nn.Module):
	def __init__(self, T_model, S_model, priorbox, channel_size, feature_num, top_k, device):
		super(KDGAN_Feat, self).__init__()
		self.T = T_model
		self.S = S_model
		self.priorbox = priorbox
		self.extract = roi_extract_multiscale(channel_size, device)
		self.feature_num = feature_num
		self.dis = nn.ModuleList()
		self.detect = Detect(21, 0, top_k, 0.01, 0.45, device)
		self.scale = [38.0, 19.0, 10.0, 5.0, 3.0, 1.0]
		self.roi_size = [7, 7, 7, 2, 2, 1]
		self.softmax = nn.Softmax(dim=-1)
		self.roi_pooling = roi_pooling_2d
		self.device = device
		for i in range(feature_num):
			self.dis.append(Discriminator(channel_size, voc['num_classes']))

	# @torchsnooper.snoop()
	def forward(self, images):
		T_feature, out_T = self.T(images)
		S_feature, out_S = self.S(images)
		priors = self.priorbox.module()
		loc = out_T[0].to(self.device)
		cls = out_T[1].to(self.device)

		out_T_test = self.detect(loc, self.softmax(cls), priors)

		out_feat_T, out_feat_S, roi = self.extract(T_feature, S_feature, out_T_test)

		d_T = []
		d_S = []
		for featT, featS, size, scale, dis in zip(out_feat_T, out_feat_S, self.roi_size[:self.feature_num], self.scale[:self.feature_num], self.dis):
			out7_T = Variable(self.roi_pooling(featT, roi, output_size=(size, size), spatial_scale=scale))
			out7_S = Variable(self.roi_pooling(featS, roi, output_size=(size, size), spatial_scale=scale), requires_grad=True)

			d_dis_t, d_cls_t, d_loc_t = dis(out7_T)
			d_dis_s, d_cls_s, d_loc_s = dis(out7_S)

			d_dis_t, d_cls_t, d_loc_t = d_dis_t.to(self.device), d_cls_t.to(self.device), d_loc_t.to(self.device)
			d_dis_s, d_cls_s, d_loc_s = d_dis_s.to(self.device), d_cls_s.to(self.device), d_loc_s.to(self.device)
			d_T.append([d_dis_t, d_cls_t, d_loc_t])
			d_S.append([d_dis_s, d_cls_s, d_loc_s])

		return d_T, d_S, out_T_test, out_S, out_T


