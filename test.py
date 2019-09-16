"""Run testing given a trained model."""

import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision
import torch.backends.cudnn as cudnn

from dataset import CoviarDataSet
from model import Model
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from transforms import GroupScale
from utils import *
import pickle
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--representation', type=str, choices=['iframe', 'residual', 'mv'])
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')
parser.add_argument('--data-root', type=str)
parser.add_argument('--test-list', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', type=str)
parser.add_argument('--save-scores', type=str, default=None)
parser.add_argument('--num-segments', type=int, default=5)
parser.add_argument('--test-crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of workers for data loader.')
parser.add_argument('--gpus', nargs='+', type=int, default=None)

args = parser.parse_args()


def main():
    writter = SummaryWriter('./log/test', comment='')

    net = Model(2, args.num_segments, args.representation,
                base_model=args.arch)

    checkpoint = torch.load(args.weights)
    # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    print("model epoch {} lowest loss {}".format(checkpoint['epoch'], checkpoint['loss_min']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.crop_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.crop_size, net.scale_size, is_mv=(args.representation == 'mv'))
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported, but got {}.".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            video_list=args.test_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=cropping,
            is_train=False,
            accumulate=(not args.no_accumulation),
        ),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

    devices = [torch.device("cuda:%d" % device) for device in args.gpus]
    net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net.eval()

    total_num = len(data_loader.dataset)
    scores = []
    labels = []
    proc_start_time = time.time()
    correct_nums = 0

    for i, (input_pairs, label) in enumerate(data_loader):
        with torch.no_grad:
            input_pairs[0] = input_pairs[0].float().to(devices[0])
            input_pairs[1] = input_pairs[1].float().to(devices[0])
            label = label.float().to(devices[0])

            outputs, y = net(input_pairs)
            _, predicts = torch.max(y, 1)
            scores.append(y.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            correct_nums += (predicts == label.clone().long()).sum()

            cnt_time = time.time() - proc_start_time
            if (i + 1) % 100 == 0:
                print('video {} done, total {}/{}, average {} sec/video'.format(i, i + 1,
                                                                                total_num,
                                                                                float(cnt_time) / (i + 1)))
    predits = np.argmax(scores, 1)
    labels = np.around(labels).astype(np.long).ravel()

    acc = 100 * correct_nums / len(data_loader.dataset)
    target_names = ['Copy', 'Not Copy']
    # writter.add_pr_curve('Precision/Recall', labels, predits)
    writter.add_text('Accuracy', '%.3f%%' % acc)
    writter.add_text(classification_report(labels, predits, target_names=target_names))
    print(('Validating Results: accuracy: {accuracy:.3f}%'.format(accuracy=acc)))

    if args.save_scores is not None:
        with open(args.save_scores + '_scores.pkl', 'wb') as fp:
            pickle.dump(scores, fp)
        with open(args.save_scores + '_labels.pkl', 'wb') as fp:
            pickle.dump(labels, fp)


if __name__ == '__main__':
    main()
