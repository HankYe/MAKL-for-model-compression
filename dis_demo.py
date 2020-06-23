import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.autograd import Variable
from layers import *
from data import VOC_CLASSES as labels
from matplotlib import pyplot as plt
from data import VOCDetection, VOCAnnotationTransform
from layers.functions.detect import match_process
from oldfiles.channelpool import ChannelPool
from oldfiles.roi_generator import roi_gen
from roi_pooling.functions.roi_pooling import roi_pooling_2d
from utils.transform import transform
from oldfiles.KDGAN_V2 import prepare_device


labels = (*labels, 'nan')
top_k = 10
devices = [0]
device, device_ids = prepare_device(devices)
othreshold = 0.3
cthreshold = 0.3
try:
    testset = VOCDetection('/home/share/Dataset/VOCdevkit/', [('2007', 'trainval')], None, VOCAnnotationTransform())
except FileNotFoundError:
    testset = VOCDetection('/home/hanchengye/data/VOCdevkit/', [('2007', 'trainval')], None, VOCAnnotationTransform())
T_model = torch.load('model/ssd300_VOC_2.pkl', map_location={'cuda:1':str(device)})
T_model = T_model.to(device)
discriminator = torch.load('model/ssdd300_VOC_2.pkl', map_location={'cuda:1':str(device)})
# param_D = torch.load('Root')
# discriminator.load_state_dict(param_D)
discriminator = discriminator.to(device)
detect = Detect(21, 0, top_k, 0.01, 0.45, device)
softmax = nn.Softmax(dim=-1)
transform = transform(1, 21)
ROI_gen = roi_gen()
ChannelPool = ChannelPool(64)
Tensor = torch.cuda.FloatTensor


def get_processed_img(img_id):
    """
    :param img_id: 图像的id
    :return: T_model的input, 给plt的input, target: [num_obj, loc+cls]
    """
    image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, ann = testset.pull_anno(img_id)
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.uint8)
    x = x[:, :, ::-1].copy()
    x = x.astype(np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.to(device)
    x = Tensor(x.unsqueeze(0))
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2).to(device)
    target = 0
    for box in ann:
        if isinstance(target, int):
            target = torch.Tensor(box).to(device)
            target[:-1] /= scale
            target = target.unsqueeze(0)
        else:
            row = torch.Tensor(box).to(device)
            row[:-1] /= scale
            row = row.unsqueeze(0)
            target = torch.cat((target, row), 0)
    target = [target]
    return x, rgb_image, target, scale

def get_processed_img2(image):
    """
    :param img_id: 图像的id
    :return: T_model的input, 给plt的input, target: [num_obj, loc+cls]
    """
    # image = testset.pull_image(img_id)
    rgb_image = image
    ann = [0, 0, 1, 1]
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.uint8)
    x = x[:, :, ::-1].copy()
    x = x.astype(np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.to(device)
    x = Tensor(x.unsqueeze(0))
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2).to(device)
    target = 0
    for box in ann:
        if isinstance(target, int):
            target = torch.Tensor(box).to(device)
            target[:-1] /= scale
            target = target.unsqueeze(0)
        else:
            row = torch.Tensor(box).to(device)
            row[:-1] /= scale
            row = row.unsqueeze(0)
            target = torch.cat((target, row), 0)
    target = [target]
    return x, rgb_image, target, scale

def get_matched_boxes(images, othreshold, cthreshold, target):
    _, T_feature, out_T = T_model(images)
    out_T_test = detect(out_T[0], softmax(out_T[1]), out_T[2])  # [1, num_cls, top_k, 5]
    # print(out_T_test)
    T_ROI, T_score = ROI_gen(out_T_test)
    T_feature = ChannelPool(T_feature)
    out7_T = roi_pooling_2d(Variable(T_feature, requires_grad=False),
                            Variable(T_ROI, requires_grad=False),
                            output_size=(7, 7), spatial_scale=8)
    input_d_T = Variable(transform(out7_T), requires_grad=False)
    cls_match, loc_match, mask, prev_loc, num_match = match_process(out_T_test, target, othreshold, cthreshold,
                                                                    device)

    out_T_test_cls = [] # [match] out_T_test对应标签
    j = 0
    for i in range(sum(mask)):
        j = list(mask).index(1, j+1)
        k = j // top_k - 1
        out_T_test_cls.append(k)
    if num_match == 0:
        raise Exception("no boxes matched")
    cls_match = cls_match.view(-1, 1)
    cls_match = cls_match.long()
    cls_match = cls_match.squeeze(1)[mask]     # [match]
    loc_match = loc_match.view(-1, 4)[mask]    # [match, 4]
    out_T_test = out_T_test.view(-1, 5)[mask]  # [match, 5]
    feature_t = input_d_T.view(-1, input_d_T.size(-3), input_d_T.size(-2), input_d_T.size(-1))[mask]
    _, d_cls_t, d_loc_t = discriminator(feature_t)  # [match, 21], [match, 4]
    d_cls_t = softmax(d_cls_t)
    d_conf, d_cls = d_cls_t.max(1, keepdim=True)
    d_cls = d_cls.squeeze(1)  # [match]
    d_loc = d_loc_t + out_T_test[:, 1:]
    d_loc[d_loc < 0] = 0
    d_loc[d_loc > 1] = 1
    return d_conf, d_cls, d_loc, cls_match, loc_match, out_T_test, out_T_test_cls


def write_detections(detections, detections_cls, scale):
    """
    detections: [num_match, conf+loc]
    detections_cls: [num_match]
    """
    color_test = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()[0]
    currentAxis = plt.gca()
    for i in range(detections.size(0)):
        score = detections[i,0]
        label_name = labels[detections_cls[i]]
        display_txt = '%s: %.2f' % (label_name, score)
        pt = (detections[i, 1:] * scale).cpu().detach().numpy()
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color_test, linewidth=1))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color_test, 'alpha': 0.5})


def write_gts(gts, scale):
    """
    gts:  [num_objs, loc+cls]
    """
    color_gt = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()[3]
    currentAxis = plt.gca()
    for i in range(gts.size(0)):
        label_idx = gts[i, 4]
        label_name = labels[int(label_idx)]
        pt = (gts[i, :4] * scale).cpu().detach().numpy()
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color_gt, linewidth=1))
        currentAxis.text(pt[0], pt[1], label_name, bbox={'facecolor': color_gt, 'alpha': 0.5})


def write_discriminator(d_conf, d_loc, d_cls, scale):
    """
    d_loc: [num_match, 4]
    d_cls: [num_match]
    """
    color_dis = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()[5]
    currentAxis = plt.gca()
    for i in range(d_cls.size(0)):
        score = d_conf[i]
        dis_idx = d_cls[i]
        label_name = labels[int(dis_idx)]
        display_txt = '%s: %.2f' % (label_name, score)
        pt = (d_loc[i, :] * scale).cpu().detach().numpy()
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color_dis, linewidth=1))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color_dis, 'alpha': 0.5})


def write_all_detections(detections, scale):
    """
    detections: [1, num_cls, top_k, conf+loc]
    """
    color_test = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()[0]
    currentAxis = plt.gca()
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= cthreshold:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().detach().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color_test, linewidth=1))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color_test, 'alpha': 0.5})
            j += 1


def get_result_1(img_id):
    """
    输入序号返回检测框与所有gt
    """
    img, rgb_image, target, scale = get_processed_img(img_id)
    d_conf, d_cls, d_loc, cls_match, loc_match, detections, detections_cls = get_matched_boxes(img, othreshold, cthreshold, target)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    # scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2).to(device)
    gts = target[0]
    write_detections(detections, detections_cls, scale)
    write_gts(gts, scale)
    plt.show()


def get_result_2(img_id):
    """
    输入序号返回检测框与discriminator回归后的检测框
    """
    img, rgb_image, target, scale = get_processed_img(img_id)
    d_conf, d_cls, d_loc, cls_match, loc_match, detections, detections_cls = get_matched_boxes(img, othreshold, cthreshold, target)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    # scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2).to(device)
    write_detections(detections, detections_cls, scale)
    write_discriminator(d_conf, d_loc, d_cls, scale)
    plt.show()


def get_result_3(img_id):
    """
    输入序号返回检测框与discriminator回归后的检测框与gtbox
    """
    img, rgb_image, target, scale = get_processed_img(img_id)
    d_conf, d_cls, d_loc, cls_match, loc_match, detections, detections_cls = get_matched_boxes(img, othreshold, cthreshold, target)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    # scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    gts = target[0]
    write_detections(detections, detections_cls, scale)
    write_discriminator(d_conf, d_loc, d_cls, scale)
    write_gts(gts, scale)
    plt.show()


def get_result_4(img_id):
    """
    输入序号所有的检测框与所有gtbox
    """
    img, rgb_image, target, scale = get_processed_img(img_id)
    T_feature, out_T = T_model(img)
    detections = detect(out_T[0], softmax(out_T[1]), out_T[2])
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    # scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2).to(device)
    gts = target[0]
    write_all_detections(detections, scale)
    write_gts(gts, scale)
    plt.show()


def get_result(img_id):
    """
    (1,1)匹配检测框(红色）和所有gtbox（黄色） (1,2)匹配检测框与对应discriminator回归结果(绿色)
    (2,1)三者都有 (2,2)所有检测框和所有gtbox
    """
    img, rgb_image, target, scale = get_processed_img(img_id)
    d_conf, d_cls, d_loc, cls_match, loc_match, detections, detections_cls = get_matched_boxes(img, othreshold, cthreshold, target)

    plt.subplot(2, 2, 1)
    plt.imshow(rgb_image)
    # scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    gts = target[0]
    write_detections(detections, detections_cls, scale)
    # write_gts(gts, scale)

    plt.subplot(2, 2, 2)
    plt.imshow(rgb_image)
    write_detections(detections, detections_cls, scale)
    write_discriminator(d_conf, d_loc, d_cls, scale)

    plt.subplot(2, 2, 3)
    plt.imshow(rgb_image)
    gts = target[0]
    write_detections(detections, detections_cls, scale)
    write_discriminator(d_conf, d_loc, d_cls, scale)
    write_gts(gts, scale)

    plt.subplot(2, 2, 4)
    plt.imshow(rgb_image)
    gts = target[0]
    _, T_feature, out_T = T_model(img)
    detections = detect(out_T[0], softmax(out_T[1]), out_T[2])  # [1, num_cls, top_k, 5]
    write_all_detections(detections, scale)
    write_gts(gts, scale)
    plt.show()

def get_result2(image):
    """
    (1,1)匹配检测框(红色）和所有gtbox（黄色） (1,2)匹配检测框与对应discriminator回归结果(绿色)
    (2,1)三者都有 (2,2)所有检测框和所有gtbox
    """
    img, rgb_image, target, scale = get_processed_img2(image)
    d_conf, d_cls, d_loc, cls_match, loc_match, detections, detections_cls = get_matched_boxes(img, othreshold, cthreshold, target)

    plt.subplot(2, 2, 1)
    plt.imshow(rgb_image)
    # scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    gts = target[0]
    write_detections(detections, detections_cls, scale)
    write_gts(gts, scale)

    plt.subplot(2, 2, 2)
    plt.imshow(rgb_image)
    write_detections(detections, detections_cls, scale)
    write_discriminator(d_conf, d_loc, d_cls, scale)

    plt.subplot(2, 2, 3)
    plt.imshow(rgb_image)
    gts = target[0]
    write_detections(detections, detections_cls, scale)
    write_discriminator(d_conf, d_loc, d_cls, scale)
    write_gts(gts, scale)

    plt.subplot(2, 2, 4)
    plt.imshow(rgb_image)
    gts = target[0]
    T_feature, out_T = T_model(img)
    detections = detect(out_T[0], softmax(out_T[1]), out_T[2])  # [1, num_cls, top_k, 5]
    write_all_detections(detections, scale)
    write_gts(gts, scale)
    plt.show()

for i in range(58, 78):
    try:
        get_result(i)
    except Exception:
        continue

# image = Image.open('')
# get_result(100)