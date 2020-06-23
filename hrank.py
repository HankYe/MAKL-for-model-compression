import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import copy
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from data import VOCDetection
from utils.augmentations import SSDAugmentation
from data import *
import torchsnooper

def prepare_device(device):
    n_gpu_use = len(device)
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
            n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    list_ids = device
    device = torch.device('cuda:{}'.format(
        device[0]) if n_gpu_use > 0 else 'cpu')

    return device, list_ids


class HRank(nn.Module):
    def __init__(self, model, limit, dataloader=None, use_gpu=True, s_threshold=None, device=0):
        """
        limit: limit of the amount of batches sent into the model for calculating average feature ranks
               e.g. send totally limit * batch_size images into the model
        # s_threshold: ignore the singular values smaller than s_threshold when calculating rank
                       default: S.max() * max(width, height) * eps  eps about 10^-7 when float32
        """
        super(HRank, self).__init__()
        self.model = model
        self.limit = limit
        self.dataloader = dataloader
        self.use_gpu = use_gpu
        self.feature_result = torch.tensor(0.).to(device)
        self.total = torch.tensor(0.).to(device)
        self.s_threshold = s_threshold
        self.device = device
        self.result = []
        if self.use_gpu:
            self.model = self.model.cuda().to(device)

    def get_feature_hook(self, module, input, output):
        a = output.size(0)
        b = output.size(1)
        # w = output.size(2)
        # h = output.size(3)
        # u, s, v = torch.svd(output.view(-1, w, h), compute_uv=False)  # s: [batch*channel, singular_values]
        # print(s[0])
        # s = torch.abs(a)
        # if self.s_threshold:
        #     s[torch.abs(s) < self.s_threshold] = 0
        # else:
        #     self.s_threshold = (10**-7) * max(w, h) * s[:,0]
        #     for i in range(int(s.size(0))):
        #         s[i][s[i] < self.s_threshold[i]] = 0
        # s = (torch.abs(s) > 0).view(a, b, s.size(-1))   # [batch, channel, singular_values]
        # s = s.sum(1).squeeze().float() / a   # [channel]
        c = torch.tensor([torch.matrix_rank(output[i, j, :, :]) for i in range(a) for j in range(b)]).to(self.device)
        c = c.view(a, -1).float()
        c = c.sum(0)
        self.feature_result = self.feature_result * self.total + c
        self.total = self.total + a
        self.feature_result = self.feature_result / self.total

    def run_data(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.dataloader):
                if batch_idx == self.limit:
                    break
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

# vgg: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512, 'M', 1024, 1024] total:35
    def forward(self, to_prune, compress_rate=0):
        """
        to_prune: integers from 0 to 14 rank filters of certain convolution layer
        compress_rate: prune compress_rate*100% filters of certain layer or the whole vgg network
        return to_prune_list [('module.vgg.to_prune.weight.filter_id', average_rank), ...]
        """
        layers = self.model.vgg
        conv_id = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 33]
        # print('ranking...')
        self.result = []
        conv_layer = layers[conv_id[to_prune]]
        handler = conv_layer.register_forward_hook(self.get_feature_hook)
        self.run_data()
        handler.remove()
        self.feature_result = list(self.feature_result.squeeze())
        for id, rank in enumerate(self.feature_result):
            self.result.append(('module.vgg.%d.weight.%d' % (conv_id[to_prune], id), float(rank)))
        self.result = sorted(self.result, key=lambda a: a[1])
        to_prune_num = int((compress_rate) * len(self.result))
        to_prune_list = [item[0] for id, item in enumerate(self.result) if id < to_prune_num]
        self.feature_result = torch.tensor(0.).to(self.device)
        self.total = torch.tensor(0.).to(self.device)

        return to_prune_list

if __name__ == '__main__':
    devices = [0,1]
    device, device_ids = prepare_device(devices)

    T_model = build_ssd("train", 300, 21)

    dataset = VOCDetection(root='/home/share/Dataset/VOCdevkit/',
                           image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                           transform=SSDAugmentation(300, MEANS))
    data_loader_rank = data.DataLoader(dataset, batch_size=1,
                                       num_workers=4, shuffle=True,
                                       collate_fn=detection_collate,
                                       pin_memory=True)
    param_T = torch.load('weights/ssd300_COCO_395000.pth', map_location={'cuda':str(device)})
    T_model.load_state_dict(param_T)
    T_model = T_model.to(device)
    # T_model = nn.DataParallel(T_model, device_ids=device_ids, output_device=device)
    criterion = HRank(T_model, 4, data_loader_rank, device=device)
    for i in range(15):
        to_prune_list = criterion(to_prune=i, compress_rate=1)
        print(len(to_prune_list))
        print(to_prune_list)
