import setproctitle
from ssd_for_KDGAN import build_ssd
from data import *
from data import VOCDetection
from utils.augmentations import SSDAugmentation
from layers.functions.detect_for_KDGAN import DiscriminatorLoss
from Prune import prune
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn import init
from layers import *
import torch
from pathlib import Path
import warnings
import visdom
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3"
viz = visdom.Visdom()
warnings.filterwarnings('ignore')
# torch.set_printoptions(profile="full")
# import tensorboard_logger as tb_logger

from kdgan_channelpool import KDGAN_Feat
from hrank import HRank

setproctitle.setproctitle("yhc:KD-TCONST")
num_cls = 21
Batch_size = 16
Channel = 128
Feature_num = 1
Epoch = 50
top_k = 100
cthreshold = 0.1
othreshold = 0.2
# logger = tb_logger.Logger(logdir='./save', flush_secs=2)
compression_rate = 5
lr_steps = (10, 20, 30)



def get_parameter_number(model):
    total_num = sum(param.numel() for param in model.parameters())
    trainable_num = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_num, trainable_num


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


def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 6)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = optimizer.param_groups[0]['lr'] * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_vis_plot(iteration, loc_kd, conf_kd, loc_s, conf_s, window1, update_type, epoch_size=1):
    viz.line(
        X=torch.ones((1, 6)).cpu() * iteration,
        Y=torch.Tensor([loc_kd, conf_kd, loc_kd + conf_kd, loc_s, conf_s, loc_s + conf_s]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )

def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if __name__ == '__main__':

    devices = [0, 1]
    device, device_ids = prepare_device(devices)

    T_file = 'model/GAN_' + repr(Feature_num) + 'F' + repr(Feature_num) + 'D/T_const/prune_list/' + repr(
            Epoch) + 'epoch/lr_102030/20200406/'
    if os.path.exists(Path(T_file)) is False:
        os.makedirs(T_file)

    priorbox = PriorBox(voc, device).to(device)
    priorbox = torch.nn.DataParallel(priorbox, device_ids=device_ids)
    T_model = build_ssd('train', 300, 21, device)

    try:
        dataset = VOCDetection(root='/home/hanchengye/data/VOCdevkit/',
                               image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                               transform=SSDAugmentation(300, MEANS))
    except FileNotFoundError:
        dataset = VOCDetection(root='/remote-home/source/remote_desktop/share/Dataset/VOCdevkit/',
                               image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                               transform=SSDAugmentation(300, MEANS))

    data_loader_rank = data.DataLoader(dataset, batch_size=32,
                                       num_workers=4, shuffle=True,
                                       collate_fn=detection_collate,
                                       pin_memory=True)

    param_T = torch.load('weights/ssd300_COCO_395000.pth')
    T_model.load_state_dict(param_T)
    T_model = T_model.to(device)
    S_model = copy.deepcopy(T_model)
    # calculate the number of the parameters in T_model
    total_param, _ = get_parameter_number(T_model)

    criterion_kd = KDloss(21).cuda()
    criterion_s = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5, False, True).cuda().to('cuda:1')
    # criterion_kd = torch.nn.DataParallel(criterion_kd, device_ids=device_ids)
    criterion_d = DiscriminatorLoss(Batch_size, num_cls, cthreshold, othreshold, top_k, device, True).cuda()
    # criterion_d = torch.nn.DataParallel(criterion_d, device_ids=device_ids)

    iteration = 0
    rate = 1.0
    while rate > 0.5:
        for p in T_model.parameters():
            p.requires_grad = False

        S_model.eval()
        criterion_rank = HRank(S_model, 16, data_loader_rank, device=device)
        criterion_rank = criterion_rank.cuda()
        criterion_rank = criterion_rank.to(device)

        print('Start ranking...')

        # toPruneList = list()
        # for i in range(15):
        #     to_prune_list = criterion_rank(to_prune=i, compress_rate=compression_rate * 0.01)
        #     print(to_prune_list)
        #     toPruneList.extend(to_prune_list)

        # toPruneList = ['module.vgg.0.weight.59', 'module.vgg.0.weight.25', 'module.vgg.0.weight.5', 'module.vgg.0.weight.22',
        #  'module.vgg.0.weight.16', 'module.vgg.0.weight.36', 'module.vgg.2.weight.2', 'module.vgg.2.weight.45',
        #  'module.vgg.2.weight.27', 'module.vgg.2.weight.18', 'module.vgg.2.weight.25', 'module.vgg.2.weight.52',
        #  'module.vgg.5.weight.69', 'module.vgg.5.weight.123', 'module.vgg.5.weight.49', 'module.vgg.5.weight.115',
        #  'module.vgg.5.weight.39', 'module.vgg.5.weight.0', 'module.vgg.5.weight.51', 'module.vgg.5.weight.105',
        #  'module.vgg.5.weight.60', 'module.vgg.5.weight.53', 'module.vgg.5.weight.28', 'module.vgg.5.weight.50',
        #  'module.vgg.7.weight.116', 'module.vgg.7.weight.127', 'module.vgg.7.weight.94', 'module.vgg.7.weight.115',
        #  'module.vgg.7.weight.61', 'module.vgg.7.weight.66', 'module.vgg.7.weight.51', 'module.vgg.7.weight.6',
        #  'module.vgg.7.weight.62', 'module.vgg.7.weight.65', 'module.vgg.7.weight.33', 'module.vgg.7.weight.119',
        #  'module.vgg.10.weight.170', 'module.vgg.10.weight.162', 'module.vgg.10.weight.184', 'module.vgg.10.weight.161',
        #  'module.vgg.10.weight.152', 'module.vgg.10.weight.239', 'module.vgg.10.weight.94', 'module.vgg.10.weight.44',
        #  'module.vgg.10.weight.90', 'module.vgg.10.weight.64', 'module.vgg.10.weight.67', 'module.vgg.10.weight.0',
        #  'module.vgg.10.weight.63', 'module.vgg.10.weight.4', 'module.vgg.10.weight.76', 'module.vgg.10.weight.149',
        #  'module.vgg.10.weight.57', 'module.vgg.10.weight.137', 'module.vgg.10.weight.178', 'module.vgg.10.weight.99',
        #  'module.vgg.10.weight.82', 'module.vgg.10.weight.15', 'module.vgg.10.weight.83', 'module.vgg.10.weight.194',
        #  'module.vgg.10.weight.120', 'module.vgg.12.weight.69', 'module.vgg.12.weight.66', 'module.vgg.12.weight.82',
        #  'module.vgg.12.weight.162', 'module.vgg.12.weight.16', 'module.vgg.12.weight.247', 'module.vgg.12.weight.125',
        #  'module.vgg.12.weight.220', 'module.vgg.12.weight.25', 'module.vgg.12.weight.165', 'module.vgg.12.weight.96',
        #  'module.vgg.12.weight.89', 'module.vgg.12.weight.213', 'module.vgg.12.weight.31', 'module.vgg.12.weight.136',
        #  'module.vgg.12.weight.85', 'module.vgg.12.weight.135', 'module.vgg.12.weight.215', 'module.vgg.12.weight.177',
        #  'module.vgg.12.weight.245', 'module.vgg.12.weight.68', 'module.vgg.12.weight.197', 'module.vgg.12.weight.216',
        #  'module.vgg.12.weight.102', 'module.vgg.12.weight.180', 'module.vgg.14.weight.133', 'module.vgg.14.weight.73',
        #  'module.vgg.14.weight.230', 'module.vgg.14.weight.129', 'module.vgg.14.weight.65', 'module.vgg.14.weight.124',
        #  'module.vgg.14.weight.144', 'module.vgg.14.weight.176', 'module.vgg.14.weight.151', 'module.vgg.14.weight.3',
        #  'module.vgg.14.weight.120', 'module.vgg.14.weight.194', 'module.vgg.14.weight.64', 'module.vgg.14.weight.178',
        #  'module.vgg.14.weight.95', 'module.vgg.14.weight.75', 'module.vgg.14.weight.34', 'module.vgg.14.weight.5',
        #  'module.vgg.14.weight.237', 'module.vgg.14.weight.207', 'module.vgg.14.weight.103', 'module.vgg.14.weight.247',
        #  'module.vgg.14.weight.250', 'module.vgg.14.weight.89', 'module.vgg.14.weight.163', 'module.vgg.17.weight.104',
        #  'module.vgg.17.weight.291', 'module.vgg.17.weight.245', 'module.vgg.17.weight.80', 'module.vgg.17.weight.364',
        #  'module.vgg.17.weight.301', 'module.vgg.17.weight.384', 'module.vgg.17.weight.89', 'module.vgg.17.weight.84',
        #  'module.vgg.17.weight.275', 'module.vgg.17.weight.100', 'module.vgg.17.weight.389', 'module.vgg.17.weight.424',
        #  'module.vgg.17.weight.240', 'module.vgg.17.weight.298', 'module.vgg.17.weight.32', 'module.vgg.17.weight.419',
        #  'module.vgg.17.weight.388', 'module.vgg.17.weight.287', 'module.vgg.17.weight.93', 'module.vgg.17.weight.257',
        #  'module.vgg.17.weight.146', 'module.vgg.17.weight.292', 'module.vgg.17.weight.403', 'module.vgg.17.weight.374',
        #  'module.vgg.17.weight.65', 'module.vgg.17.weight.106', 'module.vgg.17.weight.26', 'module.vgg.17.weight.204',
        #  'module.vgg.17.weight.215', 'module.vgg.17.weight.314', 'module.vgg.17.weight.353', 'module.vgg.17.weight.481',
        #  'module.vgg.17.weight.400', 'module.vgg.17.weight.267', 'module.vgg.17.weight.414', 'module.vgg.17.weight.8',
        #  'module.vgg.17.weight.95', 'module.vgg.17.weight.363', 'module.vgg.17.weight.302', 'module.vgg.17.weight.153',
        #  'module.vgg.17.weight.296', 'module.vgg.17.weight.327', 'module.vgg.17.weight.182', 'module.vgg.17.weight.305',
        #  'module.vgg.17.weight.295', 'module.vgg.17.weight.505', 'module.vgg.17.weight.15', 'module.vgg.17.weight.391',
        #  'module.vgg.17.weight.206', 'module.vgg.17.weight.207', 'module.vgg.19.weight.417', 'module.vgg.19.weight.308',
        #  'module.vgg.19.weight.345', 'module.vgg.19.weight.341', 'module.vgg.19.weight.137', 'module.vgg.19.weight.152',
        #  'module.vgg.19.weight.284', 'module.vgg.19.weight.115', 'module.vgg.19.weight.192', 'module.vgg.19.weight.238',
        #  'module.vgg.19.weight.328', 'module.vgg.19.weight.40', 'module.vgg.19.weight.329', 'module.vgg.19.weight.25',
        #  'module.vgg.19.weight.338', 'module.vgg.19.weight.98', 'module.vgg.19.weight.282', 'module.vgg.19.weight.498',
        #  'module.vgg.19.weight.483', 'module.vgg.19.weight.83', 'module.vgg.19.weight.155', 'module.vgg.19.weight.379',
        #  'module.vgg.19.weight.114', 'module.vgg.19.weight.251', 'module.vgg.19.weight.380', 'module.vgg.19.weight.75',
        #  'module.vgg.19.weight.376', 'module.vgg.19.weight.299', 'module.vgg.19.weight.208', 'module.vgg.19.weight.401',
        #  'module.vgg.19.weight.277', 'module.vgg.19.weight.242', 'module.vgg.19.weight.141', 'module.vgg.19.weight.74',
        #  'module.vgg.19.weight.125', 'module.vgg.19.weight.481', 'module.vgg.19.weight.225', 'module.vgg.19.weight.128',
        #  'module.vgg.19.weight.445', 'module.vgg.19.weight.20', 'module.vgg.19.weight.484', 'module.vgg.19.weight.511',
        #  'module.vgg.19.weight.477', 'module.vgg.19.weight.0', 'module.vgg.19.weight.402', 'module.vgg.19.weight.6',
        #  'module.vgg.19.weight.99', 'module.vgg.19.weight.316', 'module.vgg.19.weight.475', 'module.vgg.19.weight.363',
        #  'module.vgg.19.weight.170', 'module.vgg.21.weight.396', 'module.vgg.21.weight.499', 'module.vgg.21.weight.164',
        #  'module.vgg.21.weight.235', 'module.vgg.21.weight.152', 'module.vgg.21.weight.415', 'module.vgg.21.weight.335',
        #  'module.vgg.21.weight.500', 'module.vgg.21.weight.214', 'module.vgg.21.weight.449', 'module.vgg.21.weight.205',
        #  'module.vgg.21.weight.92', 'module.vgg.21.weight.502', 'module.vgg.21.weight.46', 'module.vgg.21.weight.151',
        #  'module.vgg.21.weight.222', 'module.vgg.21.weight.478', 'module.vgg.21.weight.440', 'module.vgg.21.weight.281',
        #  'module.vgg.21.weight.309', 'module.vgg.21.weight.229', 'module.vgg.21.weight.193', 'module.vgg.21.weight.77',
        #  'module.vgg.21.weight.473', 'module.vgg.21.weight.194', 'module.vgg.21.weight.457', 'module.vgg.21.weight.406',
        #  'module.vgg.21.weight.80', 'module.vgg.21.weight.334', 'module.vgg.21.weight.103', 'module.vgg.21.weight.276',
        #  'module.vgg.21.weight.115', 'module.vgg.21.weight.95', 'module.vgg.21.weight.52', 'module.vgg.21.weight.132',
        #  'module.vgg.21.weight.383', 'module.vgg.21.weight.138', 'module.vgg.21.weight.192', 'module.vgg.21.weight.196',
        #  'module.vgg.21.weight.296', 'module.vgg.21.weight.85', 'module.vgg.21.weight.201', 'module.vgg.21.weight.277',
        #  'module.vgg.21.weight.433', 'module.vgg.21.weight.509', 'module.vgg.21.weight.136', 'module.vgg.21.weight.287',
        #  'module.vgg.21.weight.480', 'module.vgg.21.weight.168', 'module.vgg.21.weight.476', 'module.vgg.21.weight.348',
        #  'module.vgg.24.weight.73', 'module.vgg.24.weight.169', 'module.vgg.24.weight.419', 'module.vgg.24.weight.395',
        #  'module.vgg.24.weight.405', 'module.vgg.24.weight.125', 'module.vgg.24.weight.350', 'module.vgg.24.weight.248',
        #  'module.vgg.24.weight.456', 'module.vgg.24.weight.173', 'module.vgg.24.weight.130', 'module.vgg.24.weight.60',
        #  'module.vgg.24.weight.317', 'module.vgg.24.weight.342', 'module.vgg.24.weight.487', 'module.vgg.24.weight.312',
        #  'module.vgg.24.weight.273', 'module.vgg.24.weight.134', 'module.vgg.24.weight.417', 'module.vgg.24.weight.330',
        #  'module.vgg.24.weight.250', 'module.vgg.24.weight.167', 'module.vgg.24.weight.233', 'module.vgg.24.weight.505',
        #  'module.vgg.24.weight.178', 'module.vgg.24.weight.497', 'module.vgg.24.weight.279', 'module.vgg.24.weight.362',
        #  'module.vgg.24.weight.253', 'module.vgg.24.weight.444', 'module.vgg.24.weight.260', 'module.vgg.24.weight.284',
        #  'module.vgg.24.weight.510', 'module.vgg.24.weight.63', 'module.vgg.24.weight.205', 'module.vgg.24.weight.439',
        #  'module.vgg.24.weight.147', 'module.vgg.24.weight.210', 'module.vgg.24.weight.303', 'module.vgg.24.weight.388',
        #  'module.vgg.24.weight.94', 'module.vgg.24.weight.141', 'module.vgg.24.weight.156', 'module.vgg.24.weight.261',
        #  'module.vgg.24.weight.367', 'module.vgg.24.weight.215', 'module.vgg.24.weight.278', 'module.vgg.24.weight.310',
        #  'module.vgg.24.weight.318', 'module.vgg.24.weight.10', 'module.vgg.24.weight.33', 'module.vgg.26.weight.430',
        #  'module.vgg.26.weight.29', 'module.vgg.26.weight.223', 'module.vgg.26.weight.469', 'module.vgg.26.weight.323',
        #  'module.vgg.26.weight.389', 'module.vgg.26.weight.451', 'module.vgg.26.weight.486', 'module.vgg.26.weight.16',
        #  'module.vgg.26.weight.383', 'module.vgg.26.weight.400', 'module.vgg.26.weight.242', 'module.vgg.26.weight.135',
        #  'module.vgg.26.weight.121', 'module.vgg.26.weight.309', 'module.vgg.26.weight.365', 'module.vgg.26.weight.304',
        #  'module.vgg.26.weight.259', 'module.vgg.26.weight.250', 'module.vgg.26.weight.279', 'module.vgg.26.weight.294',
        #  'module.vgg.26.weight.35', 'module.vgg.26.weight.305', 'module.vgg.26.weight.505', 'module.vgg.26.weight.313',
        #  'module.vgg.26.weight.288', 'module.vgg.26.weight.391', 'module.vgg.26.weight.171', 'module.vgg.26.weight.175',
        #  'module.vgg.26.weight.357', 'module.vgg.26.weight.87', 'module.vgg.26.weight.248', 'module.vgg.26.weight.427',
        #  'module.vgg.26.weight.23', 'module.vgg.26.weight.48', 'module.vgg.26.weight.146', 'module.vgg.26.weight.408',
        #  'module.vgg.26.weight.27', 'module.vgg.26.weight.170', 'module.vgg.26.weight.481', 'module.vgg.26.weight.148',
        #  'module.vgg.26.weight.244', 'module.vgg.26.weight.266', 'module.vgg.26.weight.314', 'module.vgg.26.weight.407',
        #  'module.vgg.26.weight.6', 'module.vgg.26.weight.220', 'module.vgg.26.weight.435', 'module.vgg.26.weight.444',
        #  'module.vgg.26.weight.155', 'module.vgg.26.weight.237', 'module.vgg.28.weight.430', 'module.vgg.28.weight.123',
        #  'module.vgg.28.weight.230', 'module.vgg.28.weight.229', 'module.vgg.28.weight.477', 'module.vgg.28.weight.375',
        #  'module.vgg.28.weight.119', 'module.vgg.28.weight.63', 'module.vgg.28.weight.172', 'module.vgg.28.weight.140',
        #  'module.vgg.28.weight.39', 'module.vgg.28.weight.246', 'module.vgg.28.weight.271', 'module.vgg.28.weight.141',
        #  'module.vgg.28.weight.313', 'module.vgg.28.weight.96', 'module.vgg.28.weight.479', 'module.vgg.28.weight.27',
        #  'module.vgg.28.weight.217', 'module.vgg.28.weight.366', 'module.vgg.28.weight.379', 'module.vgg.28.weight.11',
        #  'module.vgg.28.weight.90', 'module.vgg.28.weight.419', 'module.vgg.28.weight.500', 'module.vgg.28.weight.54',
        #  'module.vgg.28.weight.265', 'module.vgg.28.weight.10', 'module.vgg.28.weight.64', 'module.vgg.28.weight.151',
        #  'module.vgg.28.weight.381', 'module.vgg.28.weight.476', 'module.vgg.28.weight.487', 'module.vgg.28.weight.74',
        #  'module.vgg.28.weight.94', 'module.vgg.28.weight.399', 'module.vgg.28.weight.68', 'module.vgg.28.weight.241',
        #  'module.vgg.28.weight.299', 'module.vgg.28.weight.421', 'module.vgg.28.weight.38', 'module.vgg.28.weight.182',
        #  'module.vgg.28.weight.423', 'module.vgg.28.weight.70', 'module.vgg.28.weight.158', 'module.vgg.28.weight.389',
        #  'module.vgg.28.weight.411', 'module.vgg.28.weight.170', 'module.vgg.28.weight.178', 'module.vgg.28.weight.60',
        #  'module.vgg.28.weight.126', 'module.vgg.31.weight.116', 'module.vgg.31.weight.189', 'module.vgg.31.weight.216',
        #  'module.vgg.31.weight.1010', 'module.vgg.31.weight.808', 'module.vgg.31.weight.613',
        #  'module.vgg.31.weight.118', 'module.vgg.31.weight.124', 'module.vgg.31.weight.214', 'module.vgg.31.weight.248',
        #  'module.vgg.31.weight.350', 'module.vgg.31.weight.558', 'module.vgg.31.weight.581', 'module.vgg.31.weight.670',
        #  'module.vgg.31.weight.182', 'module.vgg.31.weight.271', 'module.vgg.31.weight.378', 'module.vgg.31.weight.386',
        #  'module.vgg.31.weight.423', 'module.vgg.31.weight.452', 'module.vgg.31.weight.518', 'module.vgg.31.weight.528',
        #  'module.vgg.31.weight.700', 'module.vgg.31.weight.776', 'module.vgg.31.weight.852',
        #  'module.vgg.31.weight.1016', 'module.vgg.31.weight.40', 'module.vgg.31.weight.85', 'module.vgg.31.weight.89',
        #  'module.vgg.31.weight.175', 'module.vgg.31.weight.240', 'module.vgg.31.weight.253', 'module.vgg.31.weight.263',
        #  'module.vgg.31.weight.277', 'module.vgg.31.weight.304', 'module.vgg.31.weight.312', 'module.vgg.31.weight.317',
        #  'module.vgg.31.weight.320', 'module.vgg.31.weight.372', 'module.vgg.31.weight.493', 'module.vgg.31.weight.542',
        #  'module.vgg.31.weight.547', 'module.vgg.31.weight.602', 'module.vgg.31.weight.860', 'module.vgg.31.weight.952',
        #  'module.vgg.31.weight.974', 'module.vgg.31.weight.69', 'module.vgg.31.weight.72', 'module.vgg.31.weight.108',
        #  'module.vgg.31.weight.127', 'module.vgg.31.weight.176', 'module.vgg.31.weight.179', 'module.vgg.31.weight.230',
        #  'module.vgg.31.weight.295', 'module.vgg.31.weight.336', 'module.vgg.31.weight.399', 'module.vgg.31.weight.422',
        #  'module.vgg.31.weight.444', 'module.vgg.31.weight.454', 'module.vgg.31.weight.463', 'module.vgg.31.weight.502',
        #  'module.vgg.31.weight.550', 'module.vgg.31.weight.553', 'module.vgg.31.weight.568', 'module.vgg.31.weight.646',
        #  'module.vgg.31.weight.657', 'module.vgg.31.weight.658', 'module.vgg.31.weight.697', 'module.vgg.31.weight.763',
        #  'module.vgg.31.weight.783', 'module.vgg.31.weight.855', 'module.vgg.31.weight.962', 'module.vgg.31.weight.977',
        #  'module.vgg.31.weight.980', 'module.vgg.31.weight.1005', 'module.vgg.31.weight.878', 'module.vgg.31.weight.15',
        #  'module.vgg.31.weight.76', 'module.vgg.31.weight.92', 'module.vgg.31.weight.113', 'module.vgg.31.weight.173',
        #  'module.vgg.31.weight.232', 'module.vgg.31.weight.267', 'module.vgg.31.weight.275', 'module.vgg.31.weight.297',
        #  'module.vgg.31.weight.308', 'module.vgg.31.weight.324', 'module.vgg.31.weight.331', 'module.vgg.31.weight.397',
        #  'module.vgg.31.weight.411', 'module.vgg.31.weight.471', 'module.vgg.31.weight.517', 'module.vgg.31.weight.521',
        #  'module.vgg.31.weight.523', 'module.vgg.31.weight.595', 'module.vgg.31.weight.637', 'module.vgg.31.weight.663',
        #  'module.vgg.31.weight.673', 'module.vgg.31.weight.683', 'module.vgg.31.weight.742', 'module.vgg.31.weight.757',
        #  'module.vgg.31.weight.759', 'module.vgg.33.weight.546', 'module.vgg.33.weight.833', 'module.vgg.33.weight.236',
        #  'module.vgg.33.weight.896', 'module.vgg.33.weight.712', 'module.vgg.33.weight.57', 'module.vgg.33.weight.97',
        #  'module.vgg.33.weight.353', 'module.vgg.33.weight.718', 'module.vgg.33.weight.949',
        #  'module.vgg.33.weight.1009', 'module.vgg.33.weight.94', 'module.vgg.33.weight.106', 'module.vgg.33.weight.138',
        #  'module.vgg.33.weight.159', 'module.vgg.33.weight.178', 'module.vgg.33.weight.538', 'module.vgg.33.weight.571',
        #  'module.vgg.33.weight.737', 'module.vgg.33.weight.1002', 'module.vgg.33.weight.49', 'module.vgg.33.weight.258',
        #  'module.vgg.33.weight.305', 'module.vgg.33.weight.403', 'module.vgg.33.weight.404', 'module.vgg.33.weight.407',
        #  'module.vgg.33.weight.411', 'module.vgg.33.weight.500', 'module.vgg.33.weight.502', 'module.vgg.33.weight.591',
        #  'module.vgg.33.weight.615', 'module.vgg.33.weight.631', 'module.vgg.33.weight.640', 'module.vgg.33.weight.683',
        #  'module.vgg.33.weight.765', 'module.vgg.33.weight.769', 'module.vgg.33.weight.776', 'module.vgg.33.weight.839',
        #  'module.vgg.33.weight.850', 'module.vgg.33.weight.868', 'module.vgg.33.weight.927', 'module.vgg.33.weight.950',
        #  'module.vgg.33.weight.16', 'module.vgg.33.weight.19', 'module.vgg.33.weight.35', 'module.vgg.33.weight.107',
        #  'module.vgg.33.weight.124', 'module.vgg.33.weight.139', 'module.vgg.33.weight.176', 'module.vgg.33.weight.286',
        #  'module.vgg.33.weight.346', 'module.vgg.33.weight.392', 'module.vgg.33.weight.413', 'module.vgg.33.weight.449',
        #  'module.vgg.33.weight.524', 'module.vgg.33.weight.563', 'module.vgg.33.weight.601', 'module.vgg.33.weight.614',
        #  'module.vgg.33.weight.618', 'module.vgg.33.weight.650', 'module.vgg.33.weight.675', 'module.vgg.33.weight.762',
        #  'module.vgg.33.weight.785', 'module.vgg.33.weight.826', 'module.vgg.33.weight.834', 'module.vgg.33.weight.857',
        #  'module.vgg.33.weight.891', 'module.vgg.33.weight.982', 'module.vgg.33.weight.38', 'module.vgg.33.weight.39',
        #  'module.vgg.33.weight.52', 'module.vgg.33.weight.71', 'module.vgg.33.weight.92', 'module.vgg.33.weight.114',
        #  'module.vgg.33.weight.135', 'module.vgg.33.weight.179', 'module.vgg.33.weight.192', 'module.vgg.33.weight.217',
        #  'module.vgg.33.weight.223', 'module.vgg.33.weight.228', 'module.vgg.33.weight.239', 'module.vgg.33.weight.241',
        #  'module.vgg.33.weight.252', 'module.vgg.33.weight.290', 'module.vgg.33.weight.296', 'module.vgg.33.weight.310',
        #  'module.vgg.33.weight.317', 'module.vgg.33.weight.324', 'module.vgg.33.weight.354', 'module.vgg.33.weight.362',
        #  'module.vgg.33.weight.372', 'module.vgg.33.weight.378', 'module.vgg.33.weight.459', 'module.vgg.33.weight.486',
        #  'module.vgg.33.weight.489', 'module.vgg.33.weight.499', 'module.vgg.33.weight.514', 'module.vgg.33.weight.518',
        #  'module.vgg.33.weight.528', 'module.vgg.33.weight.550', 'module.vgg.33.weight.552', 'module.vgg.33.weight.575']

        toPruneList = ['module.vgg.0.weight.59', 'module.vgg.0.weight.25', 'module.vgg.0.weight.5', 'module.vgg.0.weight.22',
         'module.vgg.0.weight.36', 'module.vgg.0.weight.16', 'module.vgg.2.weight.2', 'module.vgg.2.weight.45',
         'module.vgg.2.weight.27', 'module.vgg.2.weight.25', 'module.vgg.2.weight.18', 'module.vgg.2.weight.49',
         'module.vgg.5.weight.69', 'module.vgg.5.weight.123', 'module.vgg.5.weight.49', 'module.vgg.5.weight.115',
         'module.vgg.5.weight.0', 'module.vgg.5.weight.51', 'module.vgg.5.weight.39', 'module.vgg.5.weight.105',
         'module.vgg.5.weight.108', 'module.vgg.5.weight.60', 'module.vgg.5.weight.53', 'module.vgg.5.weight.28',
         'module.vgg.7.weight.127', 'module.vgg.7.weight.116', 'module.vgg.7.weight.94', 'module.vgg.7.weight.115',
         'module.vgg.7.weight.65', 'module.vgg.7.weight.61', 'module.vgg.7.weight.51', 'module.vgg.7.weight.62',
         'module.vgg.7.weight.66', 'module.vgg.7.weight.6', 'module.vgg.7.weight.33', 'module.vgg.7.weight.45',
         'module.vgg.10.weight.170', 'module.vgg.10.weight.162', 'module.vgg.10.weight.184', 'module.vgg.10.weight.239',
         'module.vgg.10.weight.44', 'module.vgg.10.weight.63', 'module.vgg.10.weight.4', 'module.vgg.10.weight.152',
         'module.vgg.10.weight.149', 'module.vgg.10.weight.67', 'module.vgg.10.weight.161', 'module.vgg.10.weight.90',
         'module.vgg.10.weight.0', 'module.vgg.10.weight.76', 'module.vgg.10.weight.57', 'module.vgg.10.weight.99',
         'module.vgg.10.weight.137', 'module.vgg.10.weight.178', 'module.vgg.10.weight.94', 'module.vgg.10.weight.64',
         'module.vgg.10.weight.10', 'module.vgg.10.weight.41', 'module.vgg.10.weight.83', 'module.vgg.10.weight.141',
         'module.vgg.10.weight.242', 'module.vgg.12.weight.69', 'module.vgg.12.weight.66', 'module.vgg.12.weight.82',
         'module.vgg.12.weight.162', 'module.vgg.12.weight.247', 'module.vgg.12.weight.165', 'module.vgg.12.weight.220',
         'module.vgg.12.weight.89', 'module.vgg.12.weight.25', 'module.vgg.12.weight.125', 'module.vgg.12.weight.16',
         'module.vgg.12.weight.213', 'module.vgg.12.weight.96', 'module.vgg.12.weight.136', 'module.vgg.12.weight.31',
         'module.vgg.12.weight.135', 'module.vgg.12.weight.85', 'module.vgg.12.weight.177', 'module.vgg.12.weight.215',
         'module.vgg.12.weight.180', 'module.vgg.12.weight.68', 'module.vgg.12.weight.187', 'module.vgg.12.weight.245',
         'module.vgg.12.weight.79', 'module.vgg.12.weight.241', 'module.vgg.14.weight.133', 'module.vgg.14.weight.230',
         'module.vgg.14.weight.129', 'module.vgg.14.weight.144', 'module.vgg.14.weight.73', 'module.vgg.14.weight.95',
         'module.vgg.14.weight.176', 'module.vgg.14.weight.65', 'module.vgg.14.weight.34', 'module.vgg.14.weight.124',
         'module.vgg.14.weight.3', 'module.vgg.14.weight.151', 'module.vgg.14.weight.120', 'module.vgg.14.weight.178',
         'module.vgg.14.weight.194', 'module.vgg.14.weight.75', 'module.vgg.14.weight.103', 'module.vgg.14.weight.160',
         'module.vgg.14.weight.89', 'module.vgg.14.weight.5', 'module.vgg.14.weight.169', 'module.vgg.14.weight.250',
         'module.vgg.14.weight.10', 'module.vgg.14.weight.25', 'module.vgg.14.weight.247', 'module.vgg.17.weight.291',
         'module.vgg.17.weight.301', 'module.vgg.17.weight.364', 'module.vgg.17.weight.80', 'module.vgg.17.weight.245',
         'module.vgg.17.weight.298', 'module.vgg.17.weight.89', 'module.vgg.17.weight.384', 'module.vgg.17.weight.104',
         'module.vgg.17.weight.275', 'module.vgg.17.weight.389', 'module.vgg.17.weight.84', 'module.vgg.17.weight.419',
         'module.vgg.17.weight.424', 'module.vgg.17.weight.400', 'module.vgg.17.weight.100', 'module.vgg.17.weight.403',
         'module.vgg.17.weight.146', 'module.vgg.17.weight.204', 'module.vgg.17.weight.240', 'module.vgg.17.weight.65',
         'module.vgg.17.weight.215', 'module.vgg.17.weight.257', 'module.vgg.17.weight.388', 'module.vgg.17.weight.93',
         'module.vgg.17.weight.374', 'module.vgg.17.weight.314', 'module.vgg.17.weight.414', 'module.vgg.17.weight.292',
         'module.vgg.17.weight.32', 'module.vgg.17.weight.327', 'module.vgg.17.weight.295', 'module.vgg.17.weight.26',
         'module.vgg.17.weight.353', 'module.vgg.17.weight.106', 'module.vgg.17.weight.287', 'module.vgg.17.weight.363',
         'module.vgg.17.weight.463', 'module.vgg.17.weight.505', 'module.vgg.17.weight.182', 'module.vgg.17.weight.267',
         'module.vgg.17.weight.228', 'module.vgg.17.weight.438', 'module.vgg.17.weight.296', 'module.vgg.17.weight.300',
         'module.vgg.17.weight.481', 'module.vgg.17.weight.471', 'module.vgg.17.weight.488', 'module.vgg.17.weight.302',
         'module.vgg.17.weight.305', 'module.vgg.17.weight.8', 'module.vgg.19.weight.417', 'module.vgg.19.weight.345',
         'module.vgg.19.weight.308', 'module.vgg.19.weight.341', 'module.vgg.19.weight.137', 'module.vgg.19.weight.192',
         'module.vgg.19.weight.152', 'module.vgg.19.weight.115', 'module.vgg.19.weight.40', 'module.vgg.19.weight.284',
         'module.vgg.19.weight.238', 'module.vgg.19.weight.328', 'module.vgg.19.weight.282', 'module.vgg.19.weight.329',
         'module.vgg.19.weight.379', 'module.vgg.19.weight.277', 'module.vgg.19.weight.483', 'module.vgg.19.weight.498',
         'module.vgg.19.weight.25', 'module.vgg.19.weight.338', 'module.vgg.19.weight.155', 'module.vgg.19.weight.83',
         'module.vgg.19.weight.114', 'module.vgg.19.weight.299', 'module.vgg.19.weight.98', 'module.vgg.19.weight.481',
         'module.vgg.19.weight.75', 'module.vgg.19.weight.141', 'module.vgg.19.weight.376', 'module.vgg.19.weight.251',
         'module.vgg.19.weight.20', 'module.vgg.19.weight.125', 'module.vgg.19.weight.484', 'module.vgg.19.weight.401',
         'module.vgg.19.weight.437', 'module.vgg.19.weight.225', 'module.vgg.19.weight.380', 'module.vgg.19.weight.242',
         'module.vgg.19.weight.134', 'module.vgg.19.weight.6', 'module.vgg.19.weight.147', 'module.vgg.19.weight.511',
         'module.vgg.19.weight.74', 'module.vgg.19.weight.170', 'module.vgg.19.weight.402', 'module.vgg.19.weight.477',
         'module.vgg.19.weight.208', 'module.vgg.19.weight.363', 'module.vgg.19.weight.128', 'module.vgg.19.weight.504',
         'module.vgg.19.weight.0', 'module.vgg.21.weight.396', 'module.vgg.21.weight.499', 'module.vgg.21.weight.164',
         'module.vgg.21.weight.235', 'module.vgg.21.weight.214', 'module.vgg.21.weight.415', 'module.vgg.21.weight.152',
         'module.vgg.21.weight.335', 'module.vgg.21.weight.92', 'module.vgg.21.weight.500', 'module.vgg.21.weight.222',
         'module.vgg.21.weight.478', 'module.vgg.21.weight.449', 'module.vgg.21.weight.309', 'module.vgg.21.weight.281',
         'module.vgg.21.weight.205', 'module.vgg.21.weight.46', 'module.vgg.21.weight.440', 'module.vgg.21.weight.80',
         'module.vgg.21.weight.502', 'module.vgg.21.weight.77', 'module.vgg.21.weight.229', 'module.vgg.21.weight.103',
         'module.vgg.21.weight.194', 'module.vgg.21.weight.151', 'module.vgg.21.weight.276', 'module.vgg.21.weight.473',
         'module.vgg.21.weight.115', 'module.vgg.21.weight.457', 'module.vgg.21.weight.480', 'module.vgg.21.weight.192',
         'module.vgg.21.weight.193', 'module.vgg.21.weight.383', 'module.vgg.21.weight.406', 'module.vgg.21.weight.138',
         'module.vgg.21.weight.85', 'module.vgg.21.weight.52', 'module.vgg.21.weight.136', 'module.vgg.21.weight.287',
         'module.vgg.21.weight.277', 'module.vgg.21.weight.168', 'module.vgg.21.weight.95', 'module.vgg.21.weight.476',
         'module.vgg.21.weight.132', 'module.vgg.21.weight.348', 'module.vgg.21.weight.433', 'module.vgg.21.weight.334',
         'module.vgg.21.weight.196', 'module.vgg.21.weight.201', 'module.vgg.21.weight.405', 'module.vgg.21.weight.1',
         'module.vgg.24.weight.73', 'module.vgg.24.weight.169', 'module.vgg.24.weight.419', 'module.vgg.24.weight.125',
         'module.vgg.24.weight.395', 'module.vgg.24.weight.248', 'module.vgg.24.weight.405', 'module.vgg.24.weight.456',
         'module.vgg.24.weight.350', 'module.vgg.24.weight.173', 'module.vgg.24.weight.233', 'module.vgg.24.weight.273',
         'module.vgg.24.weight.130', 'module.vgg.24.weight.342', 'module.vgg.24.weight.312', 'module.vgg.24.weight.317',
         'module.vgg.24.weight.417', 'module.vgg.24.weight.178', 'module.vgg.24.weight.487', 'module.vgg.24.weight.284',
         'module.vgg.24.weight.279', 'module.vgg.24.weight.115', 'module.vgg.24.weight.388', 'module.vgg.24.weight.60',
         'module.vgg.24.weight.505', 'module.vgg.24.weight.210', 'module.vgg.24.weight.134', 'module.vgg.24.weight.439',
         'module.vgg.24.weight.250', 'module.vgg.24.weight.330', 'module.vgg.24.weight.227', 'module.vgg.24.weight.253',
         'module.vgg.24.weight.367', 'module.vgg.24.weight.205', 'module.vgg.24.weight.376', 'module.vgg.24.weight.225',
         'module.vgg.24.weight.337', 'module.vgg.24.weight.283', 'module.vgg.24.weight.362', 'module.vgg.24.weight.147',
         'module.vgg.24.weight.63', 'module.vgg.24.weight.215', 'module.vgg.24.weight.240', 'module.vgg.24.weight.260',
         'module.vgg.24.weight.292', 'module.vgg.24.weight.22', 'module.vgg.24.weight.261', 'module.vgg.24.weight.120',
         'module.vgg.24.weight.141', 'module.vgg.24.weight.167', 'module.vgg.24.weight.94', 'module.vgg.26.weight.29',
         'module.vgg.26.weight.223', 'module.vgg.26.weight.430', 'module.vgg.26.weight.383', 'module.vgg.26.weight.389',
         'module.vgg.26.weight.400', 'module.vgg.26.weight.469', 'module.vgg.26.weight.340', 'module.vgg.26.weight.391',
         'module.vgg.26.weight.242', 'module.vgg.26.weight.323', 'module.vgg.26.weight.486', 'module.vgg.26.weight.505',
         'module.vgg.26.weight.288', 'module.vgg.26.weight.240', 'module.vgg.26.weight.427', 'module.vgg.26.weight.279',
         'module.vgg.26.weight.16', 'module.vgg.26.weight.27', 'module.vgg.26.weight.451', 'module.vgg.26.weight.117',
         'module.vgg.26.weight.121', 'module.vgg.26.weight.138', 'module.vgg.26.weight.266', 'module.vgg.26.weight.171',
         'module.vgg.26.weight.87', 'module.vgg.26.weight.309', 'module.vgg.26.weight.365', 'module.vgg.26.weight.135',
         'module.vgg.26.weight.305', 'module.vgg.26.weight.408', 'module.vgg.26.weight.146', 'module.vgg.26.weight.48',
         'module.vgg.26.weight.313', 'module.vgg.26.weight.35', 'module.vgg.26.weight.129', 'module.vgg.26.weight.248',
         'module.vgg.26.weight.250', 'module.vgg.26.weight.464', 'module.vgg.26.weight.155', 'module.vgg.26.weight.481',
         'module.vgg.26.weight.237', 'module.vgg.26.weight.259', 'module.vgg.26.weight.170', 'module.vgg.26.weight.42',
         'module.vgg.26.weight.128', 'module.vgg.26.weight.203', 'module.vgg.26.weight.220', 'module.vgg.26.weight.328',
         'module.vgg.26.weight.6', 'module.vgg.26.weight.141', 'module.vgg.28.weight.430', 'module.vgg.28.weight.123',
         'module.vgg.28.weight.477',
         'module.vgg.28.weight.230', 'module.vgg.28.weight.229', 'module.vgg.28.weight.172', 'module.vgg.28.weight.119',
         'module.vgg.28.weight.63', 'module.vgg.28.weight.375', 'module.vgg.28.weight.246', 'module.vgg.28.weight.141',
         'module.vgg.28.weight.140', 'module.vgg.28.weight.39', 'module.vgg.28.weight.479', 'module.vgg.28.weight.271',
         'module.vgg.28.weight.432', 'module.vgg.28.weight.500', 'module.vgg.28.weight.379', 'module.vgg.28.weight.25',
         'module.vgg.28.weight.217', 'module.vgg.28.weight.241', 'module.vgg.28.weight.265', 'module.vgg.28.weight.344',
         'module.vgg.28.weight.423', 'module.vgg.28.weight.38', 'module.vgg.28.weight.54', 'module.vgg.28.weight.11',
         'module.vgg.28.weight.90', 'module.vgg.28.weight.162', 'module.vgg.28.weight.251', 'module.vgg.28.weight.429',
         'module.vgg.28.weight.27', 'module.vgg.28.weight.151', 'module.vgg.28.weight.348', 'module.vgg.28.weight.419',
         'module.vgg.28.weight.68', 'module.vgg.28.weight.70', 'module.vgg.28.weight.476', 'module.vgg.28.weight.96',
         'module.vgg.28.weight.201', 'module.vgg.28.weight.299', 'module.vgg.28.weight.138', 'module.vgg.28.weight.381',
         'module.vgg.28.weight.460', 'module.vgg.28.weight.437', 'module.vgg.28.weight.12', 'module.vgg.28.weight.366',
         'module.vgg.28.weight.308', 'module.vgg.28.weight.374', 'module.vgg.28.weight.421', 'module.vgg.28.weight.158',
         'module.vgg.31.weight.528', 'module.vgg.31.weight.308', 'module.vgg.31.weight.670', 'module.vgg.31.weight.613',
         'module.vgg.31.weight.216', 'module.vgg.31.weight.240', 'module.vgg.31.weight.320', 'module.vgg.31.weight.7',
         'module.vgg.31.weight.179', 'module.vgg.31.weight.199', 'module.vgg.31.weight.808', 'module.vgg.31.weight.801',
         'module.vgg.31.weight.31', 'module.vgg.31.weight.189', 'module.vgg.31.weight.372', 'module.vgg.31.weight.517',
         'module.vgg.31.weight.959', 'module.vgg.31.weight.558', 'module.vgg.31.weight.118', 'module.vgg.31.weight.295',
         'module.vgg.31.weight.307', 'module.vgg.31.weight.333', 'module.vgg.31.weight.360', 'module.vgg.31.weight.422',
         'module.vgg.31.weight.655', 'module.vgg.31.weight.697', 'module.vgg.31.weight.912', 'module.vgg.31.weight.974',
         'module.vgg.31.weight.999', 'module.vgg.31.weight.596', 'module.vgg.31.weight.85', 'module.vgg.31.weight.87',
         'module.vgg.31.weight.152', 'module.vgg.31.weight.378', 'module.vgg.31.weight.443', 'module.vgg.31.weight.479',
         'module.vgg.31.weight.486', 'module.vgg.31.weight.778', 'module.vgg.31.weight.898', 'module.vgg.31.weight.950',
         'module.vgg.31.weight.471', 'module.vgg.31.weight.1', 'module.vgg.31.weight.54', 'module.vgg.31.weight.95',
         'module.vgg.31.weight.124', 'module.vgg.31.weight.142', 'module.vgg.31.weight.180', 'module.vgg.31.weight.271',
         'module.vgg.31.weight.472', 'module.vgg.31.weight.578', 'module.vgg.31.weight.767', 'module.vgg.31.weight.783',
         'module.vgg.31.weight.943', 'module.vgg.31.weight.986', 'module.vgg.31.weight.1010',
         'module.vgg.31.weight.234', 'module.vgg.31.weight.45', 'module.vgg.31.weight.55', 'module.vgg.31.weight.69',
         'module.vgg.31.weight.115', 'module.vgg.31.weight.136', 'module.vgg.31.weight.160', 'module.vgg.31.weight.214',
         'module.vgg.31.weight.248', 'module.vgg.31.weight.260', 'module.vgg.31.weight.264', 'module.vgg.31.weight.288',
         'module.vgg.31.weight.300', 'module.vgg.31.weight.408', 'module.vgg.31.weight.439', 'module.vgg.31.weight.581',
         'module.vgg.31.weight.653', 'module.vgg.31.weight.683', 'module.vgg.31.weight.725', 'module.vgg.31.weight.728',
         'module.vgg.31.weight.732', 'module.vgg.31.weight.740', 'module.vgg.31.weight.744', 'module.vgg.31.weight.884',
         'module.vgg.31.weight.908', 'module.vgg.31.weight.932', 'module.vgg.31.weight.949', 'module.vgg.31.weight.976',
         'module.vgg.31.weight.978', 'module.vgg.31.weight.342', 'module.vgg.31.weight.353', 'module.vgg.31.weight.66',
         'module.vgg.31.weight.71', 'module.vgg.31.weight.105', 'module.vgg.31.weight.116', 'module.vgg.31.weight.134',
         'module.vgg.31.weight.167', 'module.vgg.31.weight.183', 'module.vgg.31.weight.196', 'module.vgg.31.weight.211',
         'module.vgg.31.weight.229', 'module.vgg.31.weight.275', 'module.vgg.31.weight.279', 'module.vgg.31.weight.322',
         'module.vgg.31.weight.325', 'module.vgg.31.weight.326', 'module.vgg.31.weight.341', 'module.vgg.33.weight.546',
         'module.vgg.33.weight.833', 'module.vgg.33.weight.461', 'module.vgg.33.weight.968', 'module.vgg.33.weight.591',
         'module.vgg.33.weight.403', 'module.vgg.33.weight.925', 'module.vgg.33.weight.949', 'module.vgg.33.weight.106',
         'module.vgg.33.weight.305', 'module.vgg.33.weight.411', 'module.vgg.33.weight.561', 'module.vgg.33.weight.616',
         'module.vgg.33.weight.728', 'module.vgg.33.weight.71', 'module.vgg.33.weight.167', 'module.vgg.33.weight.345',
         'module.vgg.33.weight.367', 'module.vgg.33.weight.416', 'module.vgg.33.weight.418', 'module.vgg.33.weight.442',
         'module.vgg.33.weight.486', 'module.vgg.33.weight.500', 'module.vgg.33.weight.539', 'module.vgg.33.weight.789',
         'module.vgg.33.weight.800', 'module.vgg.33.weight.809', 'module.vgg.33.weight.885', 'module.vgg.33.weight.896',
         'module.vgg.33.weight.29', 'module.vgg.33.weight.37', 'module.vgg.33.weight.57', 'module.vgg.33.weight.67',
         'module.vgg.33.weight.75', 'module.vgg.33.weight.80', 'module.vgg.33.weight.124', 'module.vgg.33.weight.178',
         'module.vgg.33.weight.203', 'module.vgg.33.weight.217', 'module.vgg.33.weight.238', 'module.vgg.33.weight.352',
         'module.vgg.33.weight.415', 'module.vgg.33.weight.463', 'module.vgg.33.weight.517', 'module.vgg.33.weight.551',
         'module.vgg.33.weight.575', 'module.vgg.33.weight.589', 'module.vgg.33.weight.605', 'module.vgg.33.weight.615',
         'module.vgg.33.weight.624', 'module.vgg.33.weight.681', 'module.vgg.33.weight.805', 'module.vgg.33.weight.811',
         'module.vgg.33.weight.812', 'module.vgg.33.weight.818', 'module.vgg.33.weight.825', 'module.vgg.33.weight.859',
         'module.vgg.33.weight.870', 'module.vgg.33.weight.960', 'module.vgg.33.weight.1011', 'module.vgg.33.weight.19',
         'module.vgg.33.weight.28', 'module.vgg.33.weight.34', 'module.vgg.33.weight.55', 'module.vgg.33.weight.69',
         'module.vgg.33.weight.90', 'module.vgg.33.weight.94', 'module.vgg.33.weight.103', 'module.vgg.33.weight.114',
         'module.vgg.33.weight.130', 'module.vgg.33.weight.191', 'module.vgg.33.weight.192', 'module.vgg.33.weight.196',
         'module.vgg.33.weight.236', 'module.vgg.33.weight.240', 'module.vgg.33.weight.256', 'module.vgg.33.weight.270',
         'module.vgg.33.weight.299', 'module.vgg.33.weight.300', 'module.vgg.33.weight.307', 'module.vgg.33.weight.310',
         'module.vgg.33.weight.321', 'module.vgg.33.weight.338', 'module.vgg.33.weight.355', 'module.vgg.33.weight.361',
         'module.vgg.33.weight.381', 'module.vgg.33.weight.455', 'module.vgg.33.weight.475', 'module.vgg.33.weight.490',
         'module.vgg.33.weight.515', 'module.vgg.33.weight.521', 'module.vgg.33.weight.528', 'module.vgg.33.weight.552',
         'module.vgg.33.weight.571', 'module.vgg.33.weight.602', 'module.vgg.33.weight.631', 'module.vgg.33.weight.653',
         'module.vgg.33.weight.655', 'module.vgg.33.weight.659', 'module.vgg.33.weight.688', 'module.vgg.33.weight.689',
         'module.vgg.33.weight.690']

        print(str(toPruneList))

        print('Start pruning...')
        S_model = prune('VGG', S_model, toPruneList, device)

        preserved_param, _ = get_parameter_number(S_model)
        rate = preserved_param / total_param
        print('Current rate:', str(rate))

        vis_title = 'GAN compression rate: ' + str(rate)
        vis_legend = ['Loc GAN Loss', 'Conf GAN Loss', 'Total GAN Loss', 'Loc S Loss', 'Conf S Loss', 'Total S Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss(T-const)', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss(T-const)', vis_title, vis_legend)

        torch.save(S_model, T_file + 'pruned_ssd300_VOC_C' + str(Channel) + '_' + repr(iteration) + '.pkl')

        S_model = S_model.to(device)
        for p in S_model.parameters():
            p.requires_grad = True
        S_model.train()

        model = KDGAN_Feat(T_model, S_model, priorbox, Channel, Feature_num, top_k, device)
        model = model.cuda().to(device)

        print('Initializing weights...')
        model.dis.apply(weights_init)

        data_loader = data.DataLoader(dataset, batch_size=Batch_size,
                                      num_workers=4, shuffle=True,
                                      collate_fn=detection_collate,
                                      pin_memory=True, drop_last=True)

        # S_model = torch.nn.DataParallel(S_model, device_ids=device_ids)

        optimizer_D_dis = torch.optim.Adam([{'params': d.dis.parameters()} for d in model.dis], lr=0.00002, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam([{'params': d.conv1.parameters()} for d in model.dis] +
                                        [{'params': d.conv2.parameters()} for d in model.dis] +
                                        [{'params': d.conv3.parameters()} for d in model.dis], lr=0.001, betas=(0.5, 0.999))
        optimizer_D_OD = torch.optim.SGD([{'params': d.cls.parameters()} for d in model.dis] +
                                      [{'params': d.offset.parameters()} for d in model.dis], lr=1e-3, momentum=0.9, weight_decay=5e-4)
        optimizer_S = torch.optim.SGD(S_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

        scheduler_D_dis = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D_dis, mode='min', factor=0.1)
        scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.1)

        model = torch.nn.DataParallel(model, device_ids=device_ids)

        print('Start Finetuning...')
        j = 0; step_index = 0
        epoch_size = len(dataset) // Batch_size
        for epoch in range(Epoch):
            loc_loss = cls_loss = loc_loss_gan = cls_loss_gan = 0
            batch_iterator = iter(data_loader)

            if epoch in lr_steps:
                step_index += 1
                adjust_learning_rate(optimizer_S, gamma=0.1, step=step_index)
                adjust_learning_rate(optimizer_D_OD, gamma=0.1, step=step_index)

            i = 0
            for batch in range(epoch_size):
                i += 1

                try:
                    images, targets = next(batch_iterator)
                except StopIteration:
                    break

                with torch.no_grad():
                    images = Variable(images.to(device))
                    targets = [Variable(ann.to(device)) for ann in targets]

                d_T, d_S, outT_test, out_S, out_T = model(images)

                d_loss, dis_loss, s_dis_loss = criterion_d(d_T, d_S, outT_test, targets)

                if not isinstance(d_loss, int): # if 0 boxes matched, d_loss = int 0
                    optimizer_D.zero_grad()
                    optimizer_D_dis.zero_grad()
                    optimizer_D_OD.zero_grad()
                    d_loss.backward(retain_graph=True)
                    optimizer_D.step()
                    optimizer_D_dis.step()
                    optimizer_D_OD.step()
                    scheduler_D.step(d_loss)
                    scheduler_D_dis.step(dis_loss)

                loss_l_s, loss_c_s = criterion_s(out_S, targets, device)
                loc_loss += loss_l_s.data.item()
                cls_loss += loss_c_s.data.item()
                loss_l, loss_c = criterion_kd(out_S, out_T)
                loc_loss_gan += loss_c.data.item()
                cls_loss_gan += loss_l.data.item()
                S_loss = s_dis_loss
                optimizer_S.zero_grad()
                S_loss.backward()
                optimizer_S.step()

                if not isinstance(d_loss, int):
                    print("[Epoch %d/%d][Batch %d/%d][D_loss %f][S_loss: %f][D_dis_loss %f][S_dis_loss %f]"
                       % (epoch+1, Epoch, i, epoch_size, d_loss, S_loss, dis_loss, s_dis_loss)
                    )
                else:
                    print("[Epoch %d/%d][Batch %d/%d][D_loss %s][S loss: %f][D_dis_loss %s][S_dis_loss %s]"
                          % (epoch+1, Epoch, i, epoch_size, 'nan', S_loss, 'nan', 'nan')
                    )

                update_vis_plot(j, loss_l.data.item(), loss_c.data.item(), loss_l_s.data.item(), loss_c_s.data.item(), iter_plot, 'append')
                j += 1
            if epoch % 5 == 4:
                torch.save(S_model, T_file + 'ssd300_VOC_C' + str(Channel) + '_' + repr(iteration) + '_Epoch' + str(
                    epoch + 1) + '.pkl')
                torch.save(KDGAN_Feat, T_file + 'KDGAN_VOC_C' + str(Channel) + '_' + repr(iteration) + '_Epoch' + str(
                    epoch + 1) + '.pkl')
            update_vis_plot(epoch, loc_loss_gan, cls_loss_gan, loc_loss, cls_loss, epoch_plot, 'append', epoch_size)
        iteration += 1