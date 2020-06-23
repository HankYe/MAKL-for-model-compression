import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from data import *
from layers import *
from layers.functions.detect_for_KDGAN import Discriminator
from extractROI import roi_extract_multiscale
from roi_pooling.functions.roi_pooling import roi_pooling_2d
from ssd_for_KDGAN import build_ssd

class FitNet(nn.Module):
	def __init__(self, T_model, S_model, feature_num, use_gan=True, reduction='sum', device='cuda:0'):
		super(FitNet, self).__init__()
		self.T = T_model
		self.S = S_model
		self.feature_num = feature_num
		self.convS = nn.ModuleList()
		self.convT = nn.ModuleList()
		self.layer = [21, 33, 1, 3, 5, 7]
		self.device = device
		self.reduction = reduction
		self.use_gan = use_gan
		if self.use_gan:
			self.dis = nn.ModuleList()

		for i in range(feature_num):
			if i <= 1:
				inchannels = self.S.vgg[self.layer[i]].out_channels
				outchannels = self.T.vgg[self.layer[i]].out_channels
				if reduction == 'sum':
					layerT = nn.ModuleList([
						nn.Conv2d(outchannels, 1, kernel_size=1, stride=1, padding=0, bias=False),
						nn.BatchNorm2d(1, affine=False)]
					)
					layerS = nn.ModuleList([
						nn.Conv2d(inchannels, 1, kernel_size=1, stride=1, padding=0, bias=False),
						nn.BatchNorm2d(1, affine=False)]
					)
					self.convT.append(layerT)
					self.convS.append(layerS)
				else:
					self.convS.append(
						nn.Conv2d(inchannels, outchannels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
					)
				if self.use_gan:
					self.dis.append(Discriminator(outchannels, voc['num_classes']))
			elif i != 5:
				inchannels = self.S.extras[self.layer[i]].out_channels
				outchannels = self.T.extras[self.layer[i]].out_channels
				if reduction == 'sum':
					layerT = nn.ModuleList([
						nn.Conv2d(outchannels, 1, kernel_size=1, stride=1, padding=0, bias=False),
						nn.BatchNorm2d(1, affine=False)]
					)
					layerS = nn.ModuleList([
						nn.Conv2d(inchannels, 1, kernel_size=1, stride=1, padding=0, bias=False),
						nn.BatchNorm2d(1, affine=False)]
					)
					self.convT.append(layerT)
					self.convS.append(layerS)
				else:
					self.convS.append(
						nn.Conv2d(inchannels, outchannels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
					)
				if self.use_gan:
					self.dis.append(Discriminator(outchannels, voc['num_classes']))
			else:
				inchannels = self.S.extras[self.layer[i]].out_channels
				outchannels = self.T.extras[self.layer[i]].out_channels
				if reduction == 'sum':
					layerT = nn.ModuleList([
						nn.Conv2d(outchannels, 1, kernel_size=1, stride=1, padding=0, bias=False),
						nn.BatchNorm2d(1, affine=False)]
					)
					layerS = nn.ModuleList([
						nn.Conv2d(inchannels, 1, kernel_size=1, stride=1, padding=0, bias=False),
						nn.BatchNorm2d(1, affine=False)]
					)
					self.convT.append(layerT)
					self.convS.append(layerS)
				else:
					self.convS.append(nn.Linear(inchannels, outchannels, bias=False))
				if self.use_gan:
					self.dis.append(Discriminator(outchannels, voc['num_classes']))

		if reduction == 'sum':
			for i in range(feature_num):
				onesT = torch.Tensor(torch.ones_like(self.convT[i][0].weight.data))
				onesS = torch.Tensor(torch.ones_like(self.convS[i][0].weight.data))
				self.convT[i][0].weight = torch.nn.Parameter(onesT)
				self.convS[i][0].weight = torch.nn.Parameter(onesS)

			for param in self.convT.parameters():
				param.requires_grad = False

			for param in self.convS.parameters():
				param.requires_grad = False

	# @torchsnooper.snoop()
	def forward(self, images):
		T_feature, out_T = self.T(images)
		S_feature, out_S = self.S(images)
		
		if self.reduction == 'sum':
			feat_S, feat_T = [], []
			for featS, featT, convS, convT in zip(S_feature, T_feature, self.convS, self.convT):
				outfeat_S = convS[1](convS[0](featS))
				outfeat_T = convT[1](convT[0](featT))
				feat_S.append(outfeat_S)
				feat_T.append(outfeat_T)

			return feat_T, feat_S, out_S, out_T
		else:
			feat_S = []
			for featS, conv in zip(S_feature, self.convS):
				outfeat_S = conv(featS)
				feat_S.append(outfeat_S)
	
			if self.use_gan:
				disT, clsT, disS, clsS = [], [], [], []
				for featS, featT, dis in zip(feat_S, T_feature, self.dis):
					dis_T, cls_T, _ = dis(featT)
					dis_S, cls_S, _ = dis(featS)
	
					disT.append(dis_T)
					disS.append(dis_S)
					clsT.append(cls_T)
					clsS.append(cls_S)
	
				return disT, disS, clsT, clsS

			return T_feature, feat_S, out_S, out_T


if __name__ == '__main__':
	T_model = build_ssd('train', 300, 21, device='cuda:0')
	model = FitNet(T_model, T_model, 2)