# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class AdaptiveMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(AdaptiveMultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictionT, predictionS, loc_t, conf_t, device=0):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_dataT, conf_dataT, priors = predictionT
        loc_dataT = loc_dataT.detach()
        conf_dataT = conf_dataT.detach()
        loc_dataS, conf_dataS, _ = predictionS
        num = loc_dataT.size(0)

        # wrap targets
        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_dataT)
        loc_pT = loc_dataT[pos_idx].view(-1, 4).to(device)
        loc_pS = loc_dataS[pos_idx].view(-1, 4).to(device)
        loc_t = loc_t[pos_idx].view(-1, 4).to(device)
        loss_lT = F.smooth_l1_loss(loc_pT, loc_t, reduction='sum')
        loss_lS = F.smooth_l1_loss(loc_pS, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_confT = conf_dataT.view(-1, self.num_classes).to(device)
        loss_cT = log_sum_exp(batch_confT) - batch_confT.gather(1, conf_t.view(-1, 1))
        batch_confS = conf_dataS.view(-1, self.num_classes).to(device)
        loss_cS = log_sum_exp(batch_confS) - batch_confS.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_cT = loss_cT.view(num, -1)  # The line added
        loss_cS = loss_cS.view(num, -1)
        loss_cT[pos] = 0  # filter out pos boxes for now
        loss_cS[pos] = 0
        loss_cT = loss_cT.view(num, -1)
        loss_cS = loss_cS.view(num, -1)
        _, loss_idxT = loss_cT.sort(1, descending=True)
        _, idx_rankT = loss_idxT.sort(1)
        _, loss_idxS = loss_cS.sort(1, descending=True)
        _, idx_rankS = loss_idxS.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        negT = idx_rankT < num_neg.expand_as(idx_rankT)
        negS = idx_rankS < num_neg.expand_as(idx_rankS)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_dataT)
        neg_idxT = negT.unsqueeze(2).expand_as(conf_dataT)
        neg_idxS = negS.unsqueeze(2).expand_as(conf_dataS)
        conf_pT = conf_dataT[(pos_idx + neg_idxT).gt(0)].view(-1, self.num_classes).to(device)
        conf_pSkd = conf_dataS[(pos_idx + neg_idxT).gt(0)].view(-1, self.num_classes).to(device)
        conf_pS = conf_dataS[(pos_idx + neg_idxS).gt(0)].view(-1, self.num_classes).to(device)
        targets_weightedT = conf_t[(pos + negT).gt(0)].to(device)
        targets_weightedS = conf_t[(pos + negS).gt(0)].to(device)
        loss_cT = F.cross_entropy(conf_pT, targets_weightedT, reduction='sum')
        loss_cS = F.cross_entropy(conf_pS, targets_weightedS, reduction='sum')

        N = num_pos.data.sum().float()
        loss_lS /= N
        loss_cS /= N
        loss_lT /= N
        loss_cT /= N
        # slice = torch.LongTensor([[0]]).cuda()
        # slice_conf = torch.cat([slice, num_neg + num_pos], dim=0)
        # slice_loc = torch.cat([slice, num_pos], dim=0)

        # loc_kdT, loc_kdS, conf_kdT, conf_kdS, loss_locT, loss_locS, loss_confT, loss_confS = [], [], [], [], [], [], [], []
        # for i in range(len(slice_conf) - 1):
        #     slice_conf[i + 1] += slice_conf[i]
        #     slice_loc[i + 1] += slice_loc[i]
        #     loc_kdT.append(
        #         loc_pT.index_select(0, torch.arange(int(slice_loc[i]), int(slice_loc[i + 1])).cuda()))
        #     conf_kdT.append(
        #         conf_pT.index_select(0, torch.arange(int(slice_conf[i]), int(slice_conf[i + 1])).cuda()))
        #     loc_kdS.append(
        #         loc_pS.index_select(0, torch.arange(int(slice_loc[i]), int(slice_loc[i + 1])).cuda()))
        #     conf_kdS.append(
        #         conf_pSkd.index_select(0, torch.arange(int(slice_conf[i]), int(slice_conf[i + 1])).cuda()))
        #     loss_locT.append(
        #         sum(sum(loss_lT.index_select(0, torch.arange(int(slice_loc[i]), int(slice_loc[i + 1])).cuda()))) / num_pos[i])
        #     loss_locS.append(
        #         sum(sum(loss_lS.index_select(0, torch.arange(int(slice_loc[i]), int(slice_loc[i + 1])).cuda()))) / num_pos[i])
        #     loss_confT.append(
        #         sum(loss_cT.index_select(0, torch.arange(int(slice_conf[i]), int(slice_conf[i + 1])).cuda())) / num_pos[i])
        #     loss_confS.append(
        #         sum(loss_cS.index_select(0, torch.arange(int(slice_conf[i]), int(slice_conf[i + 1])).cuda())) / num_pos[i])
        #
        return (loc_pT, loc_pS), (conf_pT, conf_pSkd), (loss_lT, loss_cT), (loss_lS, loss_cS)
