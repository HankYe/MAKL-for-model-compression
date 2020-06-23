import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from data import *
import time
from ssd import build_ssd
from data import VOCDetection, VOCAnnotationTransform

Tensor = torch.cuda.FloatTensor
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def prepare_device(device):
    n_gpu_use = len(device)
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    list_ids = device
    device = torch.device('cuda:{}'.format(
        device[0]) if n_gpu_use > 0 else 'cpu')


    return device, list_ids

def mkdir(path):  # 判断是否存在指定文件夹，不存在则创建
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


class FeatureVisualize(nn.Module):
    """
    input:img_id
    在当前目录生成feature/time 文件夹 里面是各conv层输出的feature
    """
    def __init__(self, model, testset, device):
        super(FeatureVisualize, self).__init__()
        self.model = model
        self.testset = testset
        self.device = device
        try:
            self.layers = self.model.module.vgg
        except:
            self.layers = self.model.vgg
        self.conv_id = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 33]
        self.conv_list = ['1_1','1_2','2_1','2_2','3_1','3_2','3_3','4_1','4_2','4_3','5_1','5_2','5_3','ex_1','ex_2']

    def get_processed_img(self, img_id):
        image = self.testset.pull_image(img_id)
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.uint8)
        x = x[:, :, ::-1].copy()
        x = x.astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = x.to(device)
        x = Tensor(x.unsqueeze(0))
        return x, image

    def get_features(self, img_id):
        x, img = self.get_processed_img(img_id)
        result = [img]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.conv_id:
                result.append(x)
        return result

    def forward(self, img_id, now):
        features = self.get_features(img_id)
        main_path = './feature/' + now
        mkdir(main_path)
        cv2.imwrite(main_path + str(img_id) + '.jpg', features[0])
        features = features[1:]
        for i, feature in enumerate(features):
            feature = feature.squeeze(0)
            feature = feature.cpu().data.numpy()
            # use sigmod to [0,1]
            feature = 1.0 / (1 + np.exp(-1 * feature))
            # to [0,255]
            feature = np.round(feature * 255)  # [channel, width, height]
            local_path = main_path + '/conv' + self.conv_list[i]
            mkdir(local_path)
            for k in range(feature.shape[0]):
                channel = feature[k]
                cv2.imwrite(local_path + '/' + str(k) + '.jpg', channel)

def get_feature_hook_s(module, inputs, outputs):
    global feature_s
    feature_s = outputs

feature_s = None
if __name__ == '__main__':
    devices = [0]
    device, device_ids = prepare_device(devices)
    # T_model = build_ssd('train', 300, 21)
    # param_T = torch.load('weights/ssd300_COCO_395000.pth')
    T_model = torch.load('model/FitNet/prune_10/250epoch/lr_150190230/20200429/T_var_alpha_var/FitNet_ssd300_VOC_0_Epoch250.pkl')
    # T_model.load_state_dict(param_T)
    T_model = T_model.to(device)
    T_model = torch.nn.DataParallel(T_model, device_ids=device_ids)
    # model = torch.load('/model/KDGAN_1F1D/T_const/prune_5/15epoch/lr_040812/20200401/pruned_ssd300_VOC_C3_0.pkl', map_location={'cuda:1': 'cuda:0'})
    testset = VOCDetection('/remote-home/source/remote_desktop/share/Dataset/VOCdevkit/',
                           [('2007', 'trainval')], None, VOCAnnotationTransform())
    writer = FeatureVisualize(T_model, testset, device)
    writer(1, 'Finetune/')

