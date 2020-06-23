import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import *
from layers.box_utils import match
from ssd_for_KDGAN import build_ssd
from data import voc as cfg
import torchsnooper


image_index = 0

class adaptiveFitNet(nn.Module):
	def __init__(self, T_model, S_model, feature_num, threshold=0.15, startfeat=1, device='cuda:0'):
		super(adaptiveFitNet, self).__init__()
		self.T = T_model
		self.S = S_model
		self.gradients = []
		self.feature_num = feature_num
		self.handle = []
		self.device = device
		self.threshold = threshold
		self.start = startfeat
		self.variance = cfg['variance']

	def save_gradient(self, grad):
		self.gradients.append(grad)

	# @torchsnooper.snoop()
	def forward(self, x):
		sources = list()
		T_feature = list()
		loc = list()
		conf = list()

		S_feature, out_S = self.S(x)
		# apply vgg up to conv4_3 relu
		for k in range(23):
			x = self.T.vgg[k](x)
			if k == 21:
				handle1 = x.register_hook(self.save_gradient)
				self.handle.append(handle1)
				T_feature.append(x)

		s = self.T.L2Norm(x)
		sources.append(s)

		# apply vgg up to fc7
		for k in range(23, len(self.T.vgg)):
			x = self.T.vgg[k](x)
			if k == len(self.T.vgg) - 2:
				T_feature.append(x)
				handle2 = x.register_hook(self.save_gradient)
				self.handle.append(handle2)
		sources.append(x)

		# apply extra layers and cache source layer outputs
		for k, v in enumerate(self.T.extras):
			a = v(x)
			x = F.relu(a, inplace=True)
			if k % 2 == 1:
				sources.append(x)
				T_feature.append(a)
				handle = a.register_hook(self.save_gradient)
				self.handle.append(handle)

		# apply multibox head to source layers
		for (x, l, c) in zip(sources, self.T.loc, self.T.conf):
			locc = l(x)
			conff = c(x)
			loc.append(locc.permute(0, 2, 3, 1).contiguous())
			conf.append(conff.permute(0, 2, 3, 1).contiguous())

		loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
		conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
		out_T = (
			loc.view(loc.size(0), -1, 4),
			conf.view(conf.size(0), -1, self.T.num_classes),
			self.T.priors
		)

		return T_feature, S_feature, out_S, out_T

	# @torchsnooper.snoop()
	def __call__(self, images, targets):
		T_feature, S_feature, out_S, out_T = self.forward(images)
		T_feature = [feature.detach() for feature in T_feature]
		loc_data, _, priors = out_T
		loc_data = loc_data.detach()

		num = loc_data.size(0)
		priors = priors[:loc_data.size(1), :]
		num_priors = priors.size(0)

		# match priors (default boxes) and ground truth boxes
		loc_t = torch.Tensor(num, num_priors, 4).cuda()
		conf_t = torch.LongTensor(num, num_priors).cuda()
		for idx in range(num):
			truths = targets[idx][:, :-1].data
			labels = targets[idx][:, -1].data
			defaults = priors.data
			match(0.5, truths, defaults, self.variance, labels, loc_t, conf_t, idx, self.device)

		one_hot = F.one_hot(conf_t, 21).cuda()
		one_hot[:,:,0] = 0.
		one_hot = one_hot.type(torch.float32).requires_grad_(True)

		one_hot = torch.sum(one_hot * out_T[1])

		loc_t = Variable(loc_t)
		conf_t = Variable(conf_t)

		self.T.zero_grad()
		one_hot.backward()
		grads_val = self.gradients

		for handle in self.handle:
			handle.remove()

		# del one_hot
		feature_weight = torch.tensor([]).cuda()
		T_feature_embedding = []
		S_feature_embedding = []
		for i in range(self.feature_num):
			weightt = torch.mean(grads_val[-i - 1 - self.start], dim=(2, 3)).unsqueeze(2).unsqueeze(2)
			camm = torch.mul(weightt, T_feature[i + self.start]).sum(dim=1).detach()
			thresh = Variable(torch.max(camm.view(camm.shape[0], -1), dim=1).values.detach() * self.threshold)
			camm = Variable(torch.clamp(camm, min=0))
			scoree = Variable(torch.mean(camm, dim=(1, 2)).unsqueeze(1))
			mask = Variable(camm > thresh.unsqueeze(1).unsqueeze(1).expand_as(camm).abs())
			num_pos = Variable(mask.sum((1, 2)))
			maskT = Variable(mask.unsqueeze(1).expand_as(T_feature[i + self.start]).contiguous())
			maskS = Variable(mask.unsqueeze(1).expand_as(S_feature[i + self.start]).contiguous())
			channelT = T_feature[i + self.start].shape[1]
			channelS = S_feature[i + self.start].shape[1]
			embeddingT = [sum([point for point in T_feature[i + self.start][j, maskT[j]].chunk(channelT)]) / channelT for j in range(num_pos.shape[0]) if num_pos[j]]
			if len(embeddingT):
				embeddingT = torch.cat(embeddingT, dim=0).detach()
				embeddingS = torch.cat([sum([point for point in S_feature[i + self.start][j, maskS[j]].chunk(channelS)]) / channelS for j in range(num_pos.shape[0]) if num_pos[j]], dim=0).requires_grad_(True)
				T_feature_embedding.append(embeddingT)
				S_feature_embedding.append(embeddingS)
				feature_weight = torch.cat([feature_weight, scoree], dim=1)
				# v_cam = cv2.resize(camm.permute(1, 2, 0).cpu().data.numpy(), images.shape[2:]).swapaxes(0, 2).swapaxes(1, 2)
				# v_cam -= np.min(v_cam, axis=(1, 2)).reshape(-1, 1, 1)
				# v_cam /= np.max(v_cam, axis=(1, 2)).reshape(-1, 1, 1)
				#
				# show_cam_on_image(images, v_cam, i)

		feature_weight = feature_weight.sum(dim=0)
		feature_weight /= feature_weight.sum(dim=0,keepdim=True)

		self.gradients.clear()
		self.handle.clear()
		return feature_weight, T_feature_embedding, S_feature_embedding, out_T, out_S, loc_t, conf_t



def show_cam_on_image(images, cam, feature_index):
	global image_index
	for i in range(images.shape[0]):
		img = images[i].permute(1, 2, 0).cpu().data.numpy()
		img = img[:, :, ::-1].astype(np.float32)
		img += (104.0, 117.0, 123.0)
		heatmap = cv2.applyColorMap(np.uint8(255 * cam[i]), cv2.COLORMAP_JET)
		heatmap = np.float32(heatmap)
		camm = heatmap + img
		camm = camm / np.max(camm)
		cv2.imwrite("attentionmap/cam" + str(image_index) + "_feature_" + str(feature_index) + ".jpg", np.uint8(255 * camm))
		image_index += 1


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	T_model = build_ssd('train', 300, 21, device='cuda:0').cuda()
	S_model = build_ssd('train', 300, 21, device='cuda:0').cuda()
	model = adaptiveFitNet(T_model, S_model, 3)
	image = torch.randn(2,3,300,300).cuda()
	target = [Variable(torch.Tensor([[0.1, 0.1, 0.5, 0.5, 2],[0.1, 0.3, 0.2, 0.6, 15]])).cuda(),
			  Variable(torch.Tensor([[0.1, 0.5, 0.5, 0.8, 1],[0.1, 0.3, 0.2, 0.6, 19]])).cuda()]
	f_w, T_feature_embedding, S_feature_embedding, out_T, out_S, loc_t, conf_t = model(image, target)
	print(f_w)
