import torch
import torch.nn as nn
from ssd import build_ssd
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import time
from newssd import L2Norm
import copy

def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def update(toprunelist, module_index, layer_index, filter_index):
    """

    toprunelist: the list of filters to be pruned, like 'vgg.0.weight.25'
    module_index: the name of the module, like 'vgg'
    layer_index: the index of layer, like 0 in 'vgg.0.weight.25'
    filter_index: the index of filter, like 25 in 'vgg.0.weight.25'
    return toprunelist: the updated list of the rest filters
    """
    if toprunelist[0].split('.')[0] == 'module':
        updated_prunelist = [i.split('.')[0] + '.' + i.split('.')[1] + '.' + i.split('.')[2] + '.'
                             + i.split('.')[3] + '.' + str(int(i.split('.')[-1]) - 1)
                             if i.split('.')[1] == module_index and
                                i.split('.')[2] == str(layer_index) and
                                int(i.split('.')[-1]) > filter_index else i
                             for i in toprunelist]
    else:
        updated_prunelist = [i.split('.')[0] + '.' + i.split('.')[1] + '.'
                             + i.split('.')[2] + '.' + str(int(i.split('.')[-1]) - 1)
                             if i.split('.')[0] == module_index and
                                i.split('.')[1] == str(layer_index) and
                                int(i.split('.')[-1]) > filter_index else i
                             for i in toprunelist]
    # print('更新后的待删除序列：' + str(updated_prunelist))
    return updated_prunelist


def prune(netype, model, toPruneList, device=0):
    if netype.lower().find('res') is not -1:
        return group_prune(model, toPruneList, device)
    else:
        return individual_prune(model, toPruneList, device)


def group_prune(model, toPruneList, device=0):

    return 0


def prune_next_conv_layer(next_conv, filter_index, device=0):
    next_new_conv = torch.nn.Conv2d(in_channels=next_conv.in_channels - 1,
                                        out_channels=next_conv.out_channels,
                                        kernel_size=next_conv.kernel_size,
                                        stride=next_conv.stride,
                                        padding=next_conv.padding,
                                        dilation=next_conv.dilation,
                                        groups=next_conv.groups,
                                        bias=(next_conv.bias is not None))
    """
    old_weights = next_conv.weight.data.cpu().numpy()
    new_weights = next_new_conv.weight.data.cpu().numpy()

    new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
    new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]

    next_new_conv.weight.data = torch.from_numpy(new_weights).to(device)

    next_new_conv.bias.data = next_conv.bias.data
    """

    old_weights = next_conv.weight.data
    new_weights = next_new_conv.weight.data

    new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
    new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]

    next_new_conv.weight.data = new_weights

    next_new_conv.bias.data = next_conv.bias.data

    return next_new_conv


def prune_conv_layer(conv, filter_index, device=0):
    new_conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels - 1,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride, padding=conv.padding,
                         dilation=conv.dilation, groups=conv.groups,
                         bias=(conv.bias is not None))

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    try:
        new_weights[:filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
        new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
    except ValueError:
        print("filter index: ", str(filter_index))
        print("new weight size: ", str(new_weights.shape))
        print("old weight size: ", str(old_weights.shape))
        new_weights[:filter_index, :, :, :] = old_weights[: filter_index, :, :, :]

    new_conv.weight.data = torch.from_numpy(new_weights).to(device)

    bias_numpy = conv.bias.data.cpu().numpy()

    bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index + 1:]
    new_conv.bias.data = torch.from_numpy(bias).to(device)

    return new_conv


def individual_prune(T_model, toPruneList, device=0):

    toprunelist = copy.deepcopy(toPruneList)
    model = copy.deepcopy(T_model)
    if hasattr(model, 'device_ids'):
        model = model.module
    model = model.cpu()
    for k, index in enumerate(toprunelist):
        # module_index, layer_index_str, _, filter_index_str = index.split('.')
        _, module_index, layer_index_str, _, filter_index_str = toprunelist[k].split('.')

        layer_index = int(layer_index_str)
        filter_index = int(filter_index_str)
        # print('toprunelist：' + str(toprunelist))
        next_conv = None
        offset = 1
        if module_index == 'vgg':
            conv = model.vgg[layer_index]
            while layer_index + offset < len(model.vgg._modules.items()):
                layer = model.vgg[layer_index + offset]
                if isinstance(layer, nn.modules.conv.Conv2d):
                    next_conv = layer
                    if layer_index != 21:
                        break
                    else:
                        gamma = model.L2Norm.gamma
                        del model.L2Norm
                        model.L2Norm = L2Norm(conv.out_channels - 1, gamma)

                        next_conv_loc = model.loc[0]
                        next_conv_conf = model.conf[0]

                        next_new_conv_loc = prune_next_conv_layer(next_conv_loc, filter_index, device)
                        next_new_conv_conf = prune_next_conv_layer(next_conv_conf, filter_index, device)

                        loc = nn.ModuleList(
                            replace_layers(model.loc, i, [0], [next_new_conv_loc]) for i, _ in enumerate(model.loc))
                        conf = nn.ModuleList(
                            replace_layers(model.conf, i, [0], [next_new_conv_conf]) for i, _ in enumerate(model.conf))

                        del model.loc, model.conf
                        model.loc = loc
                        model.conf = conf
                        break
                else:
                    next_conv = None
                offset += 1

            new_conv = prune_conv_layer(conv, filter_index, device)

            if not next_conv is None:
                next_new_conv = prune_next_conv_layer(next_conv, filter_index, device)
                vgg = nn.ModuleList(
                    replace_layers(model.vgg, i, [layer_index, layer_index + offset], [new_conv, next_new_conv]) for i, _ in enumerate(model.vgg))

                del model.vgg
                del conv
                model.vgg = vgg

            else:
                # the last conv layer of vgg module, which has impact on the extra layers, loc layers, and conf layers
                vgg = nn.ModuleList(
                    replace_layers(model.vgg, i, [layer_index], [new_conv]) for i, _ in enumerate(model.vgg))
                del model.vgg
                model.vgg = vgg


                next_conv_extras = model.extras[0]
                next_conv_loc = model.loc[1]
                next_conv_conf = model.conf[1]

                # change extras[0] layer
                next_new_conv_extras = prune_next_conv_layer(next_conv_extras, filter_index, device)

                # change loc[1] layer
                next_new_conv_loc = prune_next_conv_layer(next_conv_loc, filter_index, device)

                # change conf[1] layer
                next_new_conv_conf = prune_next_conv_layer(next_conv_conf, filter_index, device)

                # create new modules
                extras = nn.ModuleList(
                    replace_layers(model.extras, i, [0], [next_new_conv_extras]) for i, _ in enumerate(model.extras))
                loc = nn.ModuleList(
                    replace_layers(model.loc, i, [1], [next_new_conv_loc]) for i, _ in enumerate(model.loc))
                conf = nn.ModuleList(
                    replace_layers(model.conf, i, [1], [next_new_conv_conf]) for i, _ in enumerate(model.conf))

                del model.extras, model.loc, model.conf
                model.extras = extras
                model.loc = loc
                model.conf = conf
            """
            the following part updates the list of filters to be pruned, since the model has been updated.
            """
            toprunelist = update(toprunelist, module_index, layer_index, filter_index)
            # print('实际待删除的filter：' + str(toprunelist))
        elif module_index == 'extras':
            conv = model.extras[layer_index]
            new_conv = prune_conv_layer(conv, filter_index, device)
            if layer_index + 1 < len(model.extras._modules.items()):
                next_conv = model.extras[layer_index + 1]
                next_new_conv = prune_next_conv_layer(next_conv, filter_index, device)

                if layer_index % 2 ==1:
                    next_conv_loc = model.loc[(layer_index + 1) // 2 + 1]
                    next_conv_conf = model.conf[(layer_index + 1) // 2 + 1]
                    next_new_conv_loc = prune_next_conv_layer(next_conv_loc, filter_index, device)
                    next_new_conv_conf = prune_next_conv_layer(next_conv_conf, filter_index, device)

                    loc = nn.ModuleList(
                        replace_layers(model.loc, i, [(layer_index + 1) // 2 + 1], [next_new_conv_loc]) for i, _ in enumerate(model.loc))
                    conf = nn.ModuleList(
                        replace_layers(model.conf, i, [(layer_index + 1) // 2 + 1], [next_new_conv_conf]) for i, _ in enumerate(model.conf))

                    del model.loc, model.conf
                    model.loc = loc
                    model.conf = conf

                extras = nn.ModuleList(
                    replace_layers(model.extras, i, [layer_index, layer_index + 1], [new_conv, next_new_conv]) for i, _ in enumerate(model.extras))

                del model.extras
                model.extras = extras
            else:
                next_conv_loc = model.loc[(layer_index + 1) // 2 + 1]
                next_conv_conf = model.conf[(layer_index + 1) // 2 + 1]
                next_new_conv_loc = prune_next_conv_layer(next_conv_loc, filter_index, device)
                next_new_conv_conf = prune_next_conv_layer(next_conv_conf, filter_index, device)

                extras = nn.ModuleList(
                    replace_layers(model.extras, i, [layer_index], [new_conv]) for i, _ in enumerate(model.extras))
                loc = nn.ModuleList(
                    replace_layers(model.loc, i, [(layer_index + 1) // 2 + 1], [next_new_conv_loc]) for i, _ in
                      enumerate(model.loc))
                conf = nn.ModuleList(
                    replace_layers(model.conf, i, [(layer_index + 1) // 2 + 1], [next_new_conv_conf]) for i, _ in
                      enumerate(model.conf))

                del model.extras, model.loc, model.conf
                model.loc = loc
                model.conf = conf
                model.extras = extras

            toprunelist = update(toprunelist, module_index, layer_index, filter_index)
        """
        elif module_index == 'loc':
            conv = model.loc[layer_index]
            new_conv = prune_conv_layer(conv, filter_index)
            loc = nn.ModuleList(
                    replace_layers(model.loc, i, [layer_index], [new_conv]) for i, _ in
                      enumerate(model.loc))
            del model.loc
            model.loc = loc
            toprunelist = update(toprunelist, module_index, layer_index, filter_index)
        elif module_index == 'conf':
            conv = model.conf[layer_index]
            new_conv = prune_conv_layer(conv, filter_index)
            conf = nn.ModuleList(
                replace_layers(model.conf, i, [layer_index], [new_conv]) for i, _ in
                  enumerate(model.conf))
            del model.conf
            model.conf = conf
            toprunelist = update(toprunelist, module_index, layer_index, filter_index)
        elif module_index.lower().find('norm') is not -1:
        """
        # new_model = copy.deepcopy(model)
    model = model.to(device)
    return model


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


if __name__ == "__main__":
    """# toprunelist = ['vgg.21.weight.374', 'vgg.21.weight.299', 'vgg.21.weight.383', 'vgg.0.weight.41', 'vgg.21.weight.364',
     'vgg.21.weight.303', 'vgg.21.weight.211', 'vgg.21.weight.379', 'vgg.21.weight.293', 'vgg.21.weight.92',
     'vgg.21.weight.148', 'vgg.21.weight.119', 'vgg.21.weight.487', 'vgg.21.weight.153', 'vgg.21.weight.263',
     'vgg.0.weight.48', 'vgg.21.weight.173', 'vgg.21.weight.270', 'vgg.0.weight.47', 'vgg.21.weight.112',
     'vgg.21.weight.376', 'vgg.21.weight.453', 'vgg.0.weight.43', 'vgg.21.weight.392', 'vgg.0.weight.4',
     'vgg.0.weight.10', 'vgg.0.weight.16', 'vgg.0.weight.19', 'vgg.0.weight.27', 'vgg.21.weight.29',
     'vgg.21.weight.60', 'vgg.21.weight.65', 'vgg.21.weight.67', 'vgg.21.weight.73', 'vgg.21.weight.75',
     'vgg.21.weight.98', 'vgg.21.weight.106', 'vgg.21.weight.113', 'vgg.21.weight.137', 'vgg.21.weight.139',
     'vgg.21.weight.155', 'vgg.21.weight.166', 'vgg.21.weight.170', 'vgg.21.weight.184', 'vgg.21.weight.185',
     'vgg.21.weight.188', 'vgg.21.weight.218', 'vgg.21.weight.240', 'vgg.21.weight.243', 'vgg.21.weight.245',
     'vgg.21.weight.253', 'vgg.21.weight.255', 'vgg.21.weight.273', 'vgg.21.weight.281', 'vgg.21.weight.287',
     'vgg.21.weight.292', 'vgg.21.weight.296', 'vgg.21.weight.300', 'vgg.21.weight.302', 'vgg.21.weight.313',
     'vgg.21.weight.314', 'vgg.21.weight.320', 'vgg.21.weight.329', 'vgg.21.weight.343', 'vgg.21.weight.354',
     'vgg.21.weight.366', 'vgg.21.weight.369', 'vgg.21.weight.371', 'vgg.21.weight.373', 'vgg.21.weight.378',
     'vgg.21.weight.394', 'vgg.21.weight.396', 'vgg.21.weight.398', 'vgg.21.weight.406', 'vgg.21.weight.414',
     'vgg.21.weight.422', 'vgg.21.weight.436', 'vgg.21.weight.438', 'vgg.21.weight.460', 'vgg.21.weight.464',
     'vgg.21.weight.472', 'vgg.21.weight.484', 'vgg.21.weight.488', 'vgg.21.weight.499', 'vgg.21.weight.50',
     'vgg.0.weight.63', 'vgg.21.weight.117', 'vgg.21.weight.256', 'vgg.21.weight.317', 'vgg.21.weight.429',
     'vgg.21.weight.511']"""
    toprunelist= ['module.vgg.33.weight.729', 'module.vgg.26.weight.18', 'module.vgg.26.weight.142', 'module.vgg.26.weight.197', 'module.vgg.26.weight.298', 'module.vgg.21.weight.168', 'module.vgg.26.weight.100', 'module.vgg.24.weight.269', 'module.vgg.26.weight.269', 'module.vgg.24.weight.76', 'module.vgg.28.weight.370', 'module.vgg.26.weight.78', 'module.vgg.31.weight.756', 'module.vgg.21.weight.366', 'module.vgg.26.weight.111', 'module.vgg.28.weight.364', 'module.vgg.28.weight.324', 'module.vgg.26.weight.70', 'module.vgg.24.weight.210', 'module.vgg.26.weight.466', 'module.vgg.28.weight.19', 'module.vgg.24.weight.131', 'module.vgg.17.weight.163', 'module.vgg.31.weight.545', 'module.vgg.26.weight.14', 'module.vgg.28.weight.84', 'module.vgg.26.weight.179', 'module.vgg.24.weight.308', 'module.vgg.28.weight.214', 'module.vgg.28.weight.295', 'module.vgg.28.weight.355', 'module.vgg.31.weight.131', 'module.vgg.19.weight.323', 'module.vgg.33.weight.663', 'module.vgg.24.weight.252', 'module.vgg.21.weight.86', 'module.vgg.17.weight.335', 'module.vgg.21.weight.119', 'module.vgg.33.weight.11', 'module.vgg.5.weight.77', 'module.vgg.31.weight.534', 'module.vgg.24.weight.508', 'module.vgg.28.weight.136', 'module.vgg.28.weight.500', 'module.vgg.33.weight.238', 'module.vgg.33.weight.870', 'module.vgg.33.weight.1000', 'module.vgg.26.weight.176', 'module.vgg.31.weight.973', 'module.vgg.33.weight.338', 'module.vgg.33.weight.804', 'module.vgg.28.weight.377', 'module.vgg.24.weight.334', 'module.vgg.19.weight.152', 'module.vgg.33.weight.914', 'module.vgg.33.weight.718', 'module.vgg.33.weight.752', 'module.vgg.31.weight.558', 'module.vgg.31.weight.525', 'module.vgg.33.weight.600', 'module.vgg.33.weight.999', 'module.vgg.24.weight.275', 'module.vgg.17.weight.427', 'module.vgg.28.weight.290', 'module.vgg.31.weight.469', 'module.vgg.33.weight.195', 'module.vgg.31.weight.575', 'module.vgg.24.weight.505', 'module.vgg.31.weight.637', 'module.vgg.33.weight.998', 'module.vgg.33.weight.589', 'module.vgg.31.weight.347', 'module.vgg.24.weight.166', 'module.vgg.31.weight.695', 'module.vgg.33.weight.327', 'module.vgg.33.weight.884', 'module.vgg.33.weight.45', 'module.vgg.26.weight.13', 'module.vgg.26.weight.331', 'module.vgg.33.weight.367', 'module.vgg.24.weight.352', 'module.vgg.31.weight.339', 'module.vgg.33.weight.765', 'module.vgg.33.weight.818', 'module.vgg.31.weight.974', 'module.vgg.33.weight.95', 'module.vgg.33.weight.403', 'module.vgg.33.weight.38', 'module.vgg.33.weight.88', 'module.vgg.24.weight.233', 'module.vgg.28.weight.299', 'module.vgg.19.weight.189', 'module.vgg.24.weight.143', 'module.vgg.33.weight.323', 'module.vgg.33.weight.439', 'module.vgg.24.weight.288', 'module.vgg.33.weight.27', 'module.vgg.33.weight.202', 'module.extras.1.weight.414', 'module.vgg.31.weight.54', 'module.vgg.31.weight.972', 'module.vgg.33.weight.44', 'module.vgg.33.weight.146', 'module.vgg.28.weight.26', 'module.vgg.33.weight.685', 'module.vgg.21.weight.87', 'module.vgg.31.weight.328', 'module.vgg.33.weight.481', 'module.vgg.33.weight.852', 'module.extras.0.weight.135', 'module.vgg.33.weight.580', 'module.vgg.21.weight.407', 'module.vgg.31.weight.946', 'module.vgg.33.weight.796', 'module.vgg.28.weight.123', 'module.vgg.26.weight.409', 'module.vgg.28.weight.172', 'module.vgg.33.weight.509', 'module.extras.0.weight.48', 'module.extras.1.weight.235', 'module.vgg.33.weight.78', 'module.vgg.33.weight.623', 'module.vgg.33.weight.268', 'module.vgg.31.weight.514', 'module.vgg.28.weight.176', 'module.vgg.24.weight.423', 'module.vgg.26.weight.64', 'module.vgg.33.weight.266', 'module.vgg.31.weight.348', 'module.vgg.31.weight.600', 'module.vgg.33.weight.79', 'module.vgg.33.weight.794', 'module.vgg.33.weight.777', 'module.vgg.26.weight.406', 'module.vgg.33.weight.229', 'module.extras.1.weight.298', 'module.vgg.31.weight.955', 'module.vgg.31.weight.93', 'module.vgg.21.weight.207', 'module.vgg.19.weight.165', 'module.vgg.33.weight.498', 'module.extras.4.weight.14', 'module.vgg.21.weight.155', 'module.vgg.31.weight.887', 'module.vgg.33.weight.605', 'module.vgg.21.weight.463', 'module.vgg.31.weight.964', 'module.vgg.33.weight.141', 'module.vgg.31.weight.439', 'module.vgg.28.weight.332', 'module.vgg.31.weight.188', 'module.vgg.31.weight.521', 'module.vgg.33.weight.324', 'module.vgg.24.weight.130', 'module.vgg.33.weight.18', 'module.vgg.28.weight.349', 'module.vgg.31.weight.467', 'module.vgg.33.weight.721', 'module.vgg.33.weight.880', 'module.vgg.33.weight.469', 'module.vgg.33.weight.869', 'module.vgg.31.weight.717', 'module.vgg.33.weight.843', 'module.vgg.33.weight.643', 'module.vgg.24.weight.342', 'module.vgg.31.weight.325', 'module.vgg.33.weight.968', 'module.vgg.31.weight.1019', 'module.vgg.33.weight.83', 'module.vgg.33.weight.211', 'module.vgg.33.weight.234', 'module.vgg.33.weight.264', 'module.vgg.5.weight.123', 'module.vgg.28.weight.179', 'module.vgg.33.weight.664', 'module.extras.1.weight.284', 'module.vgg.31.weight.61', 'module.vgg.31.weight.536', 'module.vgg.33.weight.510', 'module.vgg.33.weight.819', 'module.vgg.21.weight.499', 'module.vgg.24.weight.101', 'module.vgg.31.weight.870', 'module.vgg.33.weight.262', 'module.vgg.31.weight.692', 'module.vgg.33.weight.40', 'module.vgg.31.weight.506', 'module.vgg.31.weight.685', 'module.vgg.33.weight.158', 'module.vgg.24.weight.391', 'module.vgg.26.weight.271', 'module.vgg.31.weight.171', 'module.extras.1.weight.63', 'module.extras.1.weight.330', 'module.extras.2.weight.72', 'module.vgg.31.weight.485', 'module.vgg.31.weight.95', 'module.vgg.24.weight.439', 'module.vgg.33.weight.609', 'module.vgg.33.weight.832', 'module.extras.4.weight.21']

    devices = [0, 1, 2, 3]
    device, device_ids = prepare_device(devices)
    # toprunelist = ['module.vgg.0.weight.19', 'module.vgg.0.weight.59', 'module.vgg.0.weight.31', 'module.vgg.0.weight.29', 'module.vgg.0.weight.16', 'module.vgg.0.weight.5', 'module.vgg.0.weight.45', 'module.vgg.0.weight.8', 'module.vgg.0.weight.28', 'module.vgg.0.weight.47', 'module.vgg.0.weight.37', 'module.vgg.0.weight.51', 'module.vgg.0.weight.14', 'module.vgg.0.weight.23', 'module.vgg.0.weight.49', 'module.vgg.0.weight.35', 'module.vgg.0.weight.46', 'module.vgg.0.weight.53', 'module.vgg.0.weight.12', 'module.vgg.0.weight.26', 'module.vgg.0.weight.25', 'module.vgg.0.weight.41', 'module.vgg.0.weight.34', 'module.vgg.0.weight.62', 'module.vgg.0.weight.21', 'module.vgg.0.weight.43', 'module.vgg.0.weight.48', 'module.vgg.0.weight.1', 'module.vgg.0.weight.32', 'module.vgg.0.weight.52', 'module.vgg.0.weight.15', 'module.vgg.0.weight.3', 'module.vgg.0.weight.7', 'module.vgg.0.weight.60', 'module.vgg.0.weight.27', 'module.vgg.0.weight.22', 'module.vgg.0.weight.10', 'module.vgg.0.weight.57', 'module.vgg.0.weight.54', 'module.vgg.0.weight.18', 'module.vgg.0.weight.33', 'module.vgg.0.weight.13', 'module.vgg.0.weight.20', 'module.vgg.0.weight.39', 'module.vgg.0.weight.36', 'module.vgg.0.weight.50', 'module.vgg.0.weight.58', 'module.vgg.0.weight.55', 'module.vgg.0.weight.44', 'module.vgg.0.weight.61', 'module.vgg.0.weight.56', 'module.vgg.0.weight.0', 'module.vgg.0.weight.9', 'module.vgg.0.weight.6', 'module.vgg.0.weight.4', 'module.vgg.0.weight.24', 'module.vgg.0.weight.38', 'module.vgg.0.weight.11']


    model = build_ssd('train', 300, 21)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = prune('vgg', model, toprunelist, device)

    a = 1