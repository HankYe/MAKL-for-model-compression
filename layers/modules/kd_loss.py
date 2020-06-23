import torch.nn as nn
from layers.modules.multibox_loss import MultiBoxLoss
import torch
class KDloss(nn.Module):
	def __init__(self, numcls=21, temperature=1.0, minlossT=(0.3, 1.3), maxlossT=(0.7, 1.7), device='cuda:0', reduction_kl='batchmean', reduction_l1='mean'):
		super(KDloss, self).__init__()
		self.numclass = numcls
		self.kldiv = nn.KLDivLoss(reduction=reduction_kl)
		self.smoothl1 = nn.SmoothL1Loss(reduction=reduction_l1)
		self.softmax = nn.Softmax(dim=-1)
		self.logsoftmax = nn.LogSoftmax(dim=-1)
		self.tau = temperature
		self.minlossT = minlossT
		self.maxlossT = maxlossT
		self.finetuneloss = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5, False, True)
		self.device = device

	def forward(self, student, teacher, target):
		locT, clsT, _ = teacher
		locS, clsS, _ = student

		soft_logS = self.logsoftmax(clsS.view(-1, self.numclass) / self.tau)
		soft_probT = self.softmax(clsT.view(-1, self.numclass) / self.tau)

		loc_lossS, cls_lossS = self.finetuneloss(student, target, self.device)
		loc_lossT, cls_lossT = self.finetuneloss(teacher, target, self.device)

		wmin_loc, wmin_cls = self.minlossT
		wmax_loc, wmax_cls = self.maxlossT
		alpha_c = float(torch.sqrt(torch.abs(1. - (cls_lossT - wmin_cls) ** 2 / (wmax_cls ** 2))))
		alpha_l = float(torch.sqrt(torch.abs(1. - (loc_lossT - wmin_loc) ** 2 / (wmax_loc ** 2))))

		loss_c = self.kldiv(soft_logS, soft_probT) * (1. - alpha_c) * self.tau * self.tau + alpha_c * cls_lossS

		loss_l = self.smoothl1(locS.view(-1, 4), locT.view(-1, 4)) * (1. - alpha_l) + alpha_l * loc_lossS

		return loss_l, loss_c, loc_lossS, cls_lossS