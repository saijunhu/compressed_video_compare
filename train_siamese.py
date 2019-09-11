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
# from dataset import CoviarDataSet
from dataset_hsj import CoviarDataSet
from model import Model
from train_options import parser
from transforms import GroupCenterCrop
from transforms import GroupScale
from utils import *
from torch.utils.tensorboard import SummaryWriter

SAVE_FREQ = 40
PRINT_FREQ = 20
ACCUMU_STEPS = 4  # use gradient accumlation to use least memory and more runtime
loss_min = 1
CONTINUE_FROM_LAST = False
LAST_SAVE_PATH = r'vcdb_mv_res50_copy_detection_checkpoint.pth.tar'
FINETUNE = False

# for visualization
writer = SummaryWriter('./log',comment='')


def main():
    global args
    global devices
    args = parser.parse_args()

    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    model = Model(2, args.num_segments, args.representation,
                  base_model=args.arch)
    # writer.add_graph(model,input_to_model=())
    # add continue train from before
    if CONTINUE_FROM_LAST:
        checkpoint = torch.load(LAST_SAVE_PATH)
        # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
        print("model epoch {} lowest loss {}".format(checkpoint['epoch'], checkpoint['loss_min']))
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        loss_min = checkpoint['loss_min']
        model.load_state_dict(base_dict)
    else:
        loss_min = 10000

    # print(model)
    # writer.add_graph(model, (torch.randn(10,5, 2, 224, 224),))

    devices = [torch.device("cuda:%d" % device) for device in args.gpus]

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.train_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=model.get_augmentation(),
            is_train=True,
            accumulate=(not args.no_accumulation),
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.test_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=torchvision.transforms.Compose([
                GroupScale(int(model.scale_size)),
                GroupCenterCrop(model.crop_size),
            ]),
            is_train=False,
            accumulate=(not args.no_accumulation),
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model = model.to(devices[0])
    cudnn.benchmark = True

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0

        if ('module.base_model.conv1' in key
            or 'module.base_model.bn1' in key
            or 'data_bn' in key) and args.representation in ['mv', 'residual']:
            lr_mult = 0.1
        elif '.fc.' in key:
            lr_mult = 1.0
        else:
            lr_mult = 0.01

        params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]

    if FINETUNE:
        optimizer = torch.optim.SGD(params, lr=1e-5, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(
            params,
            weight_decay=args.weight_decay,
            eps=0.001)

    criterions = []
    siamese_loss = ContrastiveLoss(margin=2.0).to(devices[0])
    classifiy_loss = nn.CrossEntropyLoss()
    criterions.append(siamese_loss)
    criterions.append(classifiy_loss)

    # try to use ReduceOnPlatue to adjust lr
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15 // args.eval_freq, verbose=True)

    for epoch in range(args.epochs):
        train(train_loader, model, criterions, optimizer, epoch)
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss_cur = validate(val_loader, model, criterions, epoch)
            scheduler.step(loss_cur)
            is_best = (loss_cur < loss_min)
            loss_min = min(loss_cur, loss_min)
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
    writer.close()


def train(train_loader, model, criterions, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    siamese_losses = AverageMeter()
    clf_losses = AverageMeter()

    model.train()

    end = time.time()
    correct_num = 0
    for i, (input_pairs, label) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_pairs[0] = input_pairs[0].float().to(devices[0])
        input_pairs[1] = input_pairs[1].float().to(devices[0])
        label = label.float().to(devices[0])

        outputs, y = model(input_pairs)
        loss1 = criterions[0](outputs[0], outputs[1], label.clone().float()) / ACCUMU_STEPS
        loss2 = criterions[1](y, label.clone().long()) / ACCUMU_STEPS
        siamese_losses.update(loss1.item(), input_pairs[0].size(0))
        clf_losses.update(loss2.item(), input_pairs[0].size(0))

        loss1.backward(retain_graph=True)
        loss2.backward()

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


def validate(val_loader, model, criterions, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    siamese_losses = AverageMeter()
    clf_losses = AverageMeter()

    # model.float()
    model.eval()

    end = time.time()
    correct_nums = 0

    for i, (input_pairs, label) in enumerate(val_loader):
        with torch.no_grad():
            input_pairs[0] = input_pairs[0].float().to(devices[0])
            input_pairs[1] = input_pairs[1].float().to(devices[0])
            label = label.float().to(devices[0])

            outputs, y = model(input_pairs)
            loss1 = criterions[0](outputs[0], outputs[1], label.clone().float())
            loss2 = criterions[1](y, label.clone().long())
            siamese_losses.update(loss1.item(), input_pairs[0].size(0))
            clf_losses.update(loss2.item(), input_pairs[0].size(0))

            _, predicts = torch.max(y, 1)
            correct_nums += (predicts == label.clone().long()).sum()
            batch_time.update(time.time() - end)
            end = time.time()

            # for tensorboard

            if i % PRINT_FREQ == 0:
                print(('Validate: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'siamese loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                       ' loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                       ' loss {loss2.val:.4f} ({loss2.avg:.4f})\t'
                    .format(
                    i, len(val_loader),
                    batch_time=batch_time,
                    loss1=siamese_losses,
                    loss2=clf_losses)))

    acc = 100 * correct_nums / len(val_loader.dataset)
    print(('Validating Results: siamese Loss {loss.avg:.5f}, Accuracy: {accuracy:.3f}%'.format(loss=siamese_losses,
                                                                                               accuracy=acc)))
    writer.add_scalar('Accuracy/epoch', acc, epoch)
    writer.add_scalar('Siamese Loss/epoch', siamese_losses.avg, epoch)
    writer.add_scalar('Classification Loss/epoch', clf_losses.avg, epoch)
    return siamese_losses.avg


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((args.model_prefix, filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.model_prefix, '_best.pth.tar'))
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
