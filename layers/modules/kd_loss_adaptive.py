import torch.nn as nn
from layers.modules.multibox_loss_adaptive import AdaptiveMultiBoxLoss
import torch
import torchsnooper

class AdaptiveKDloss(nn.Module):
	def __init__(self, numcls=21, temperature=1.0, beta=10, gamma=2., minlossT=(0.3, 1.3), maxlossT=(0.7, 1.7), device='cuda:0', reduction_kl='batchmean', reduction_l1='mean'):
		super(AdaptiveKDloss, self).__init__()
		self.numclass = numcls
		self.kldiv = nn.KLDivLoss(reduction=reduction_kl)
		self.smoothl1 = nn.SmoothL1Loss(reduction=reduction_l1)
		self.softmax = nn.Softmax(dim=-1)
		self.logsoftmax = nn.LogSoftmax(dim=-1)
		self.tau = temperature
		self.beta = beta
		self.gamma = gamma
		self.minlossT = minlossT
		self.maxlossT = maxlossT
		self.device = device

	# @torchsnooper.snoop()
	def forward(self, loc_kd, conf_kd, lossT, lossS):
		loc_kdT, loc_kdS = loc_kd
		conf_kdT, conf_kdS = conf_kd
		loc_lossT, cls_lossT = lossT
		loc_lossS, cls_lossS = lossS

		wmin_loc, wmin_cls = self.minlossT
		wmax_loc, wmax_cls = self.maxlossT

		# w = []
		# for i in range(len(loc_kdT)):
		# 	wi = float((1 - (self.softmax(conf_kdT[i]).view(-1) / conf_kdT[i].shape[0]).dot((self.softmax(conf_kdS[i]).view(-1) / conf_kdT[i].shape[0]))) ** self.gamma)
		# 	w.append(wi)
		# w_mean = sum(w) / len(loc_kdT)
		#
		# loss_l, loss_c, loss_locS, loss_clsS = [], [], 0., 0.
		# for i in range(len(loc_kdT)):
		tau = float(self.tau + torch.exp((wmax_cls - cls_lossT) / (wmax_cls - wmin_cls)) * self.beta)

		soft_logS = self.logsoftmax(conf_kdS.view(-1, self.numclass) / tau)
		soft_probT = self.softmax(conf_kdT.view(-1, self.numclass) / tau)

		alpha_c = float(torch.sqrt(torch.abs(1. - (cls_lossT - wmin_cls) ** 2 / ((wmax_cls - wmin_cls) ** 2))))
		alpha_l = float(torch.sqrt(torch.abs(1. - (loc_lossT - wmin_loc) ** 2 / ((wmax_loc - wmin_loc) ** 2))))
		print(tau, alpha_c, alpha_l)
		loss_c = self.kldiv(soft_logS, soft_probT) * alpha_c * tau * tau + (1. - alpha_c) * cls_lossS
		loss_l = self.smoothl1(loc_kdS.view(-1, 4), loc_kdT.view(-1, 4)) * alpha_l + (1. - alpha_l) * loc_lossS

		return loss_l, loss_c