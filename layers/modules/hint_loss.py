import torch.nn as nn
import torch

class HintLoss(nn.Module):
	def __init__(self, reduction='mean'):
		super(HintLoss, self).__init__()
		self.hint = nn.MSELoss(reduction=reduction)
		self.weights = [0.5, 0.3, 0.1, 0.05, 0.05]

	def forward(self, student, teacher):
		loss_feat_total = []
		for weight, featT, featS in zip(self.weights, teacher, student):
			loss_feat = self.hint(featS.view(featS.shape[0], -1), featT.view(featT.shape[0], -1))
			loss_feat_total.append(loss_feat * weight)

		return sum(loss_feat_total) / len(teacher)