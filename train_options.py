"""Training options."""

import argparse

parser = argparse.ArgumentParser(description="CoViAR")

# Data.
# parser.add_argument('--data-name', type=str, choices=['ucf101', 'hmdb51', 'vcdb'],
#                     help='dataset name.')
parser.add_argument('--data-root', type=str,
                    help='root of data directory.')
parser.add_argument('--train-list', type=str,
                    help='training example list.')
parser.add_argument('--test-list', type=str,
                    help='testing example list.')

# Model.
parser.add_argument('--representation', type=str, choices=['iframe', 'mv', 'residual', 'mixed'],
                    help='data representation.')
parser.add_argument('--arch', type=str, default="resnet152",
                    help='base architecture.')
parser.add_argument('--num-segments', type=int, default=3,
                    help='number of TSN segments.')
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')
# store_true 是指带触发action时为真，不触发则为假，2L说的代码去掉default初始化，其功能也不会变化
parser.add_argument('--dropout', default=0.25, type=float,
                    help='control the dropout ratio')
parser.add_argument('--keyfeatures', default=128, type=int,
                    help='the discriminative key vectors length')

# Training.
parser.add_argument('--epochs', default=500, type=int,
                    help='number of training epochs.')
parser.add_argument('--batch-size', default=40, type=int,
                    help='batch size.')
parser.add_argument('--lr', default=0.001, type=float,
                    help='base learning rate.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay.')
parser.add_argument('--gpus', nargs='+', type=int, default=None,
                    help='gpu ids.')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loader workers.')

# Log.
parser.add_argument('--eval-freq', default=5, type=int,
                    help='evaluation frequency (epochs).')

