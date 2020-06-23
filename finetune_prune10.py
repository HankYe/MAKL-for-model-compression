from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from newssd import new_build_ssd

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='/remote-home/source/remote_desktop/share/Dataset/VOCdevkit/',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='model/Finetune/prune_list/20200418/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

if args.visdom:
    import visdom
    viz = visdom.Visdom()

def train():

    T_file = 'model/Finetune/prune_list/300epoch/lr_150190230/20200418/'
    if os.path.exists(Path(T_file)) is False:
        os.makedirs(T_file)
    lr_steps = (150, 190, 230)
    cfg = voc
    dataset = VOCDetection(root=args.dataset_root, image_sets = [('2007', 'trainval'), ('2012', 'trainval')],
                           transform=SSDAugmentation(cfg['min_dim'],
                                                     MEANS))
    testset = VOCDetection(root=args.dataset_root,
                           image_sets=[('2007', 'test')],
                           transform=SSDAugmentation(300, MEANS))

    ssd_net = torch.load('model/Finetune/prune_list/300epoch/lr_150190230/20200418/finetune_ssd300_VOC_Epoch120.pkl', map_location={'cuda:0': 'cuda:0'})
    ssd_net.cuda()

    optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    ssd_net.train()

    if args.visdom:
        vis_title = 'Finetune prune_list on' + dataset.name
        test_title = 'Finetune prune_list on test07' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
        test_plot = create_vis_plot('Epoch', 'Loss(test)', test_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    data_loader_test = data.DataLoader(testset, batch_size=32, num_workers=4, shuffle=False,
                                       collate_fn=detection_collate, pin_memory=True)

    ssd_net.to('cuda:1')
    batch_test = iter(data_loader_test)
    epoch_test = len(data_loader_test) // 32
    loc_test_total, cls_test_total = 0., 0.
    for batch in range(epoch_test):
        try:
            images, targets = next(batch_test)
        except StopIteration:
            break

        with torch.no_grad():
            images = Variable(images.to('cuda:1'))
            targets = [Variable(ann.to('cuda:1')) for ann in targets]
        _, output = ssd_net(images)
        loc_test, cls_test = criterion(output, targets, 'cuda:1')
        loc_test_total += loc_test
        cls_test_total += cls_test
    if args.visdom:
        update_vis_plot(121, loc_test_total, cls_test_total, test_plot, None, 'append', epoch_test)
    best_loss = (loc_test_total + cls_test_total) / epoch_test
    print('pruned loss: %.4f' % best_loss)

    ssd_net.to('cuda:0')

    # create batch iterator
    print('Start Training...')
    j = 121
    step_index = 0
    epoch_size = len(dataset) // 32
    for epoch in range(120, 300):
        loc_loss = conf_loss = 0
        batch_iterator = iter(data_loader)

        if epoch in lr_steps:
            step_index += 1
            adjust_learning_rate(optimizer, gamma=0.1, step=step_index)

        i = 0
        for iters in range(epoch_size):
            i += 1

            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                break

            with torch.no_grad():
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            t0 = time.time()
            _, out = ssd_net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            if iters % 10 == 0:
                iteration = iters + epoch * epoch_size
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

            if args.visdom:
                update_vis_plot(j, loss_l.data.item(), loss_c.data.item(),
                                iter_plot, epoch_plot, 'append')
            j += 1

        update_vis_plot(epoch + 1, loc_loss, conf_loss, epoch_plot, None, 'append', epoch_size)
        if epoch % 5 == 4:
            torch.save(ssd_net, T_file + 'finetune_ssd300_VOC_Epoch' + str(
                epoch + 1) + '.pkl')

        ssd_net.to('cuda:1')
        batch_test = iter(data_loader_test)
        loc_test_total, cls_test_total = 0., 0.
        for batch in range(epoch_test):
            try:
                images, targets = next(batch_test)
            except StopIteration:
                break

            with torch.no_grad():
                images = Variable(images.to('cuda:1'))
                targets = [Variable(ann.to('cuda:1')) for ann in targets]
            _, output = ssd_net(images)
            loc_test, cls_test = criterion(output, targets, 'cuda:1')
            loc_test_total += loc_test.data.item()
            cls_test_total += cls_test.data.item()
        update_vis_plot(epoch + 2, loc_test_total, cls_test_total, test_plot, None, 'append', epoch_test)
        if best_loss >= (loc_test_total + cls_test_total) / epoch_test:
            best_loss = (loc_test_total + cls_test_total) / epoch_test
            print('best loss: %.4f' % best_loss)

        ssd_net.cuda()

    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
