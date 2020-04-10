"""Run training."""

import shutil
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends as F
from imqfusion.dataset_gjy import CoviarDataSet
from imqfusion.model_im_fm_stack import Model
from train_options import parser

from torch.utils.tensorboard import SummaryWriter

from utils.utils import AverageMeter, ContrastiveLoss, get_lr, WarmStartCosineAnnealingLR

SAVE_FREQ = 5
PRINT_FREQ = 20
ACCUMU_STEPS = 4  # use gradient accumlation to use least memory and more runtime
loss_min = 1
CONTINUE_FROM_LAST = False
LAST_SAVE_PATH = r'r2plus1d_18_bt_24_seg_10_ifame_fusion_qp+mv_part_dataset_checkpoint.pth.tar'
FINETUNE = False

WEI_S = 1
WEI_C = 2

# for visualization
WRITER = []
DEVICES = []

description = ""


def main():
    print(torch.cuda.device_count())
    global args
    global devices
    global WRITER
    args = parser.parse_args()
    global description
    description = 'bt_%d_seg_%d_%s' % (args.batch_size * ACCUMU_STEPS, args.num_segments, "im_fm_conv1_stack_sgd")
    log_name = './log/%s' % description
    WRITER = SummaryWriter(log_name)
    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    model = Model(2, args.num_segments, args.representation,
                  base_model=args.arch)

    # add continue train from before
    if CONTINUE_FROM_LAST:
        checkpoint = torch.load(LAST_SAVE_PATH)
        # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
        print("model epoch {} lowest loss {}".format(checkpoint['epoch'], checkpoint['loss_min']))
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        loss_min = checkpoint['loss_min']
        model.load_state_dict(base_dict)
        start_epochs = checkpoint['epoch']
    else:
        loss_min = 10000
        start_epochs = 0

    devices = [torch.device("cuda:%d" % device) for device in args.gpus]
    global DEVICES
    DEVICES = devices

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.train_list,
            num_segments=args.num_segments,
            is_train=True,
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.test_list,
            num_segments=args.num_segments,
            is_train=False,
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model = model.to(devices[0])
    cudnn.benchmark = True

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0
        if 'module.fc' in key:
            params += [{'params': [value], 'lr': args.lr * 10, 'decay_mult': decay_mult}]
        elif 'module.fusion' in key:
            params += [{'params': [value], 'lr': args.lr * 10, 'decay_mult': decay_mult}]
        elif 'module.mvnet' in key:
            params += [{'params': [value], 'lr': args.lr * 10, 'decay_mult': decay_mult}]
        else:
            params += [{'params': [value], 'lr': args.lr * 1, 'decay_mult': decay_mult}]

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    criterions = []
    siamese_loss = ContrastiveLoss(margin=2.0).to(devices[0])
    classifiy_loss = nn.CrossEntropyLoss().to(devices[0])
    # classifiy_loss = LabelSmoothingLoss(2,0.1,-1)
    criterions.append(siamese_loss)
    criterions.append(classifiy_loss)

    # try to use ReduceOnPlatue to adjust lr
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20 // args.eval_freq, verbose=True)
    scheduler = WarmStartCosineAnnealingLR(optimizer, T_max=args.epochs, T_warm=10)
    for epoch in range(start_epochs, args.epochs):
        # about optimizer
        WRITER.add_scalar('Lr/epoch', get_lr(optimizer), epoch)
        loss_train_s, loss_train_c = train(train_loader, model, criterions, optimizer, epoch)
        loss_train = WEI_S * loss_train_s + WEI_C * loss_train_c
        scheduler.step(epoch)
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss_val_s, loss_val_c, acc = validate(val_loader, model, criterions, epoch)
            loss_val = WEI_S * loss_val_s + WEI_C * loss_val_c
            is_best = (loss_val_c < loss_min)
            loss_min = min(loss_val_c, loss_min)
            # visualization
            WRITER.add_scalar('Accuracy/epoch', acc, epoch)
            WRITER.add_scalars('Siamese Loss/epoch', {'Train': loss_train_s, 'Val': loss_val_s}, epoch)
            WRITER.add_scalars('Classification Loss/epoch', {'Train': loss_train_c, 'Val': loss_val_c}, epoch)
            WRITER.add_scalars('Combine Loss/epoch', {'Train': loss_train, 'Val': loss_val}, epoch)
            if is_best or epoch % SAVE_FREQ == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'loss_min': loss_min,
                    },
                    is_best,
                    filename='checkpoint.pth.tar')
    WRITER.close()


def train(train_loader, model, criterions, optimizer, epoch):
    '''
    :param train_loader:
    :param model:
    :param criterions:
    :param optimizer:
    :param epoch:
    :return:  (siamese loss, clf loss)
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    siamese_losses = AverageMeter()
    clf_losses = AverageMeter()

    model.train()
    end = time.time()
    for i, (input_pairs, label) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_pairs[0][0] = input_pairs[0][0].float().to(devices[0])
        input_pairs[0][1] = input_pairs[0][1].float().to(devices[0])
        input_pairs[1][0] = input_pairs[1][0].float().to(devices[0])
        input_pairs[1][1] = input_pairs[1][1].float().to(devices[0])
        label = label.float().to(devices[0])
        outputs, y = model(input_pairs)
        loss1 = criterions[0](outputs[0], outputs[1], label.clone().float()) / ACCUMU_STEPS
        loss2 = criterions[1](y, label.clone().long()) / ACCUMU_STEPS
        siamese_losses.update(loss1.item(), args.batch_size)
        clf_losses.update(loss2.item(), args.batch_size)
        # loss1.backward(retain_graph=True)
        # loss2.backward()
        loss = WEI_S * loss1 + WEI_C * loss2
        loss.backward()
        # use gradient accumulation
        if i % ACCUMU_STEPS == 0:
            # attention the following line can't be transplaced
            optimizer.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}],\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'siamese Loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                   'classifier Loss {loss2.val:.4f} ({loss2.avg:.4f})\t'.format(
                epoch, i, len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss1=siamese_losses,
                loss2=clf_losses)))

    return siamese_losses.avg, clf_losses.avg  # attention indent ,there was a serious bug here


def validate(val_loader, model, criterions, epoch):
    '''
    :param val_loader:
    :param model:
    :param criterions:
    :param epoch:
    :return:  (siamese loss, clf loss, acc)
    '''
    batch_time = AverageMeter()
    siamese_losses = AverageMeter()
    clf_losses = AverageMeter()
    model.eval()
    end = time.time()
    correct_nums = 0
    for i, (input_pairs, label) in enumerate(val_loader):
        with torch.no_grad():
            input_pairs[0][0] = input_pairs[0][0].float().to(devices[0])
            input_pairs[0][1] = input_pairs[0][1].float().to(devices[0])
            input_pairs[1][0] = input_pairs[1][0].float().to(devices[0])
            input_pairs[1][1] = input_pairs[1][1].float().to(devices[0])
            label = label.float().to(devices[0])

            outputs, y = model(input_pairs)
            loss1 = criterions[0](outputs[0], outputs[1], label.clone().float())
            loss2 = criterions[1](y, label.clone().long())
            siamese_losses.update(loss1.item(), input_pairs[0][0].size(0))
            clf_losses.update(loss2.item(), input_pairs[0][0].size(0))

            _, predicts = torch.max(y, 1)
            correct_nums += (predicts == label.clone().long()).sum()
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print(('Validate: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'siamese loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                       'clf loss {loss2.val:.4f} ({loss2.avg:.4f})\t'
                    .format(
                    i, len(val_loader),
                    batch_time=batch_time,
                    loss1=siamese_losses,
                    loss2=clf_losses)))

    acc = 100 * correct_nums / len(val_loader.dataset)
    print((
        'Validating Results: siamese Loss {loss.avg:.5f}, classification loss {loss3.avg:.5f}, Accuracy: {accuracy:.3f}%'.format(
            loss=siamese_losses, loss3=clf_losses,
            accuracy=acc)))
    return siamese_losses.avg, clf_losses.avg, acc


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((description, filename))
    filename = './' + filename
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((description, '_best.pth.tar'))
        best_name = './' + best_name
        shutil.copyfile(filename, best_name)


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    wd = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr


if __name__ == '__main__':
    main()
