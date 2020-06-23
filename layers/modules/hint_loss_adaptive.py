import torch.nn as nn
import torch

class AdaptiveHintLoss(nn.Module):
	def __init__(self, reduction='mean'):
		super(AdaptiveHintLoss, self).__init__()
		self.hint = nn.MSELoss(reduction=reduction)

	def forward(self, student_embedding, teacher_embedding, weights=None):
		loss_feat_total = []
		if weights is not None:
			for weight, featT, featS in zip(weights, teacher_embedding, student_embedding):
				loss_feat_total.append(self.hint(featS, featT) * weight)
		else:
			for featT, featS in zip(teacher_embedding, student_embedding):
				loss_feat_total.append(self.hint(featS, featT))
		return sum(loss_feat_total)