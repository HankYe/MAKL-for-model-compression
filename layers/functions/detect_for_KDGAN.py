import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchsnooper
from layers.box_utils import jaccard
from layers.box_utils import match


"""
  网络test模式下的输出:
  [Batch, num_cls, top_k, score+loc(5)]
  top_k前若干个对应score+loc为预测的置信度与坐标，之后的都补0

  网络train模式下的输出：
  [Batch, priors, loc(4)], [Batch, priors, num_cls], priors
"""

def conv3x3(inchannel, outchannel, padding=0):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1,
                     padding=padding, bias=True)


def linear(insize, outsize):
    return nn.Linear(insize, outsize)


class Discriminator(nn.Module):
    """
    in_channel:输入的channel数
    cls_num:目标的类别数
    """
    def __init__(self, in_channel, cls_num):
        super(Discriminator, self).__init__()
        self.cls_num = cls_num
        # 卷积层
        self.conv1 = conv3x3(in_channel, 256)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv3x3(256, 256)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = conv3x3(256, 512, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 真假判断
        self.dis = conv3x3(512, 1)

        # 分类
        self.cls = conv3x3(512, self.cls_num)

        # 回归输出左上角右下角的坐标
        self.offset = conv3x3(512, 4)

    def forward(self, in_feats, dis=False):
        """
        :param in_feats: [Batch, channels, feature(7x7)]
        :return: [Batch, 1], [Batch, num_cls], [Batch, 4]
        """
        out = self.conv1(in_feats)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        # 真假判断及loss
        pre_dis = self.dis(out)
        pre_dis = torch.sigmoid(pre_dis)
        if dis:
            return pre_dis

        # 分类
        pre_cls = self.cls(out)

        # 回归
        pre_offset = self.offset(out)

        return pre_dis.view(pre_dis.shape[0], pre_dis.shape[1]), \
               pre_cls.view(pre_cls.shape[0], pre_cls.shape[1]), \
               pre_offset.view(pre_offset.shape[0], pre_offset.shape[1])


# @torchsnooper.snoop()
def match_dis(othreshold, cthreshold, prevs_loc, prevs_conf, truths_loc, truths_cls, loc_match, cls_match, mask, idx):
    """
    args
        othreshold: overlap threshold
        cthreshold: conf threshold
        prevs_loc: tensor [num_prevs, 4]
        prevs_conf: tensor [num_prevs]
        truths_loc: tensor [num_objs, 4]
        truths_cls: tensor [num_objs]
        loc_match: to be filled [batch, num_pres, 4]
        cls_match: to be filled [batch, num_pres]
        mask : to be filled [batch, num_pres]
        idx : current batch index
        count: num of good prevs
    return
        none
    """
    overlaps = jaccard(truths_loc, prevs_loc)  # [num_objs, num_prevs]
    # print('overlap',overlaps)

    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)  # [num_prevs] 每个prev最大overlap的obj序号
    best_truth_overlap.squeeze_(0)  # 每个prev的最大overlap
    # print('best',best_truth_idx)
    # print('xiaocl',cls_match[idx])

    # for i in range(cls_match.size(1)):
    #     cls_match[idx,i] = truths_cls[int(best_truth_idx[i])]  # [num_prevs] 每个prev匹配obj的cls
    cls_match[idx] = truths_cls[best_truth_idx]
    # print('houcl',cls_match[idx])
    loc_match[idx] = truths_loc[best_truth_idx, :]  # [num_prevs, 4] 每个prev匹配obj的loc

    mask[idx][best_truth_overlap < othreshold] = 0
    mask[idx][prevs_conf < cthreshold] = 0
    # print(prevs_loc[mask[idx]])
    # print(loc_match[:,mask[idx]])
    # print(prevs_conf[mask[idx]])
    # print(overlaps[:,mask[idx]])
    # print(count)

def match_process(prev_t, target, othreshold, cthreshold, device):
    """
    :param feature_t: [batch, num_cls, top_k, channel, width, width]
    :param feature_s: [batch, num_cls, top_k, channel, width, width]
    :param prev_t: [batch, num_cls, top_k, conf+loc(5)]
    :param target: (list) [[[xmin, ymin, xmax, ymax, label]...]...]
    :param othreshold: overlap threshold
    :param cthreshold: conf threshold
    :param device: device
    :return: cls_match:[batch, num_prevs(num_cls * top_k)]
             loc_match:[batch, num_prevs, 4]
             mask: [batch * num_prevs]
             num_matches: count how many gt-boxes were matched
    """
    batch_size = prev_t.size(0)
    num_prevs = prev_t.size(1) * prev_t.size(2)
    mask = torch.ones(batch_size, num_prevs).to(device)
    mask = mask.type(torch.bool)
    cls_match = torch.zeros(batch_size, num_prevs).to(device)
    loc_match = torch.zeros(batch_size, num_prevs, 4).to(device)
    prev_t = prev_t.view(batch_size, -1, 5)  # [batch, boxes, conf+loc(5)]
    prev_loc = prev_t[:, :, 1:]  # [batch, prevs, 4]
    prev_conf = prev_t[:, :, 0]  # [batch, prevs]
    for i in range(batch_size):
        match_dis(othreshold, cthreshold, prev_loc[i], prev_conf[i], target[i][:, :-1],
                  target[i][:, 4], loc_match, cls_match,mask, i)
    mask = mask.view(-1)
    num_match = int(torch.sum(mask))
    print('totally %d boxes matched' % (num_match))
    return cls_match, loc_match, mask, prev_loc, num_match

# @torchsnooper.snoop()
class DiscriminatorLoss(nn.Module):
    def __init__(self, batchsize, num_class, conf_thresh, overlap_thresh, top_k, device, use_gpu=True):
        super(DiscriminatorLoss, self).__init__()
        self.use_gpu = use_gpu
        self.batchsize = batchsize
        self.num_class = num_class
        self.cthreshold = conf_thresh
        self.othreshold = overlap_thresh
        self.top_k = top_k
        self.device = device
        self.num_prevs = self.num_class * self.top_k

    # @torchsnooper.snoop()
    def forward(self, d_T, d_S, prev_t, target):
        cls_match, loc_match, mask, prev_loc, num_match = match_process(prev_t, target, self.othreshold, self.cthreshold, self.device)

        if num_match == 0:
            return 0, 0, 0
        real = torch.ones(self.batchsize * self.num_prevs, 1).to(self.device)
        fake = torch.zeros(self.batchsize * self.num_prevs, 1).to(self.device)

        real = real[mask]
        fake = fake[mask]
        cls_match = cls_match.view(-1, 1)
        cls_match = cls_match.long()
        cls_match = cls_match.squeeze(1)

        prev_loc = prev_loc.view(-1, 4)
        loc_match = loc_match.view(-1, 4)

        dis_loss_total = 0
        d_loss_total = 0
        s_dis_loss_total = 0
        for d_t, d_s in zip(d_T, d_S):
            d_dis_t, d_cls_t, d_loc_t = d_t
            d_dis_s, d_cls_s, d_loc_s = d_s

            # 分类 cls_loss 比较 d_cls_s 与 cls_gt
            cls_loss = F.cross_entropy(d_cls_s[mask], cls_match[mask])

            # 回归 loc_loss
            d_loc = d_loc_t[mask] + Variable(prev_loc[mask])
            loc_loss = F.smooth_l1_loss(d_loc, loc_match[mask])

            # 判别 dis_loss
            real_loss = F.binary_cross_entropy(d_dis_t[mask], real)
            fake_loss = F.binary_cross_entropy(d_dis_s[mask], fake)
            dis_loss = 0.5 * real_loss + 0.5 * fake_loss
            dis_loss_total += dis_loss

            # student loss from dis
            s_dis_loss = F.binary_cross_entropy(d_dis_s[mask], real)
            s_dis_loss_total += s_dis_loss

            d_loss = loc_loss + cls_loss + dis_loss
            d_loss_total += d_loss
        return d_loss_total / len(d_T), dis_loss_total / len(d_T), s_dis_loss_total / len(d_T)

def test_discriminatorloss():
    discriminator = Discriminator(1, 21)
    feature_t = torch.autograd.Variable(torch.randn([2, 2, 10, 1, 7, 7])).cuda()
    feature_s = torch.autograd.Variable(torch.randn(2, 2, 10, 1, 7, 7)).cuda()
    loc_t = torch.autograd.Variable(torch.randn(2, 2, 10, 5)).cuda()
    gt = torch.FloatTensor([[0, 0, 0, 0, 3], [0.5, 0, 0.7, 0.5, 2], [0.1,0.2,0.3,0.4,7]]).cuda()
    gt = [gt,gt]
    criterion = DiscriminatorLoss(discriminator, 2, -1, -1, 10, 'cuda', True)
    d_loss, s_dis_loss = criterion(feature_t, feature_s, loc_t, gt)
    d_loss.backward()
    print(d_loss, s_dis_loss)

# test_discriminatorloss()
