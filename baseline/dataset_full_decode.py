"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors,
or residuals) for training or testing.
"""

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath) # 把项目的根目录添加到程序执行时的环境变量

import os
import os.path
import random
import math
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import traceback
import logging
import matplotlib.pyplot as plt
import torchvision
from utils.sample import *
from torchvision.io import read_video
from transforms import *

# from memory_profiler import profile
import gc
VIDEOS_URL = r'/home/sjhu/datasets/all_dataset'
## For Dataset
WIDTH = 256
HEIGHT = 340

torchvision.set_video_backend('video_reader')

class FullDecodeDataSet(data.Dataset):
    def __init__(self, data_root,
                 video_list,
                 num_segments,
                 is_train):

        self._data_root = data_root
        self._num_segments = num_segments
        self._is_train = is_train
        self._iframe_scales = [1, .875, .75]
        self._mv_scales = [1, .875, .75, .66]
        self._input_size = 224
        self._scale_size = self._input_size * 256 // 224
        self._iframe_transform = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, self._iframe_scales),
             GroupRandomHorizontalFlip(is_mv=False)])
        self._infer_transform = torchvision.transforms.Compose([
            GroupScale(int(self._scale_size)),
            GroupCenterCrop(self._input_size),
        ])
        # modify depend on Kinetics-400 dataset setting
        self._input_mean = np.array([0.43216, 0.394666, 0.37645]).reshape((3, 1, 1, 1)).astype(np.float32)
        self._input_std = np.array([0.22803, 0.22145, 0.216989]).reshape((3, 1, 1, 1)).astype(np.float32)

        self._load_list(video_list)

    def _load_list(self, video_list):
        """
        :param video_list: input the txt file contains the speific split
        :return: get a list contains video path and name
        """
        self._videos_list = []
        self._labels_list = []
        with open(video_list, 'r') as f:
            for line in f:
                # A,B,1
                video_a, video_b, label = line.strip().split(',')
                self._videos_list.append((video_a, video_b))
                self._labels_list.append(int(label))
        # must be numpy array rather than python list , nontheless ,this may be lead to a memory leak.
        self._videos_list = np.array(self._videos_list)
        self._labels_list = np.array(self._labels_list)
        self._size = len(self._labels_list)
        print('%d pair videos loaded.' % self._size)


    def __getitem__(self, index):
        # siamese label, '0' means same, '1' means diff
        video_pairs = self._videos_list[index]
        pairs_data = []  # ( img1,img2 )

        for video in video_pairs:
            # shapes  (nums,height,width,channels)
            # # process iframe
            frames,_,_ = read_video(video)
            # print(frames.shape)
            if len(frames) == 0:
                print("decode frame failed")
                frames = np.array(np.zeros((self._num_segments,256,340,3)),dtype=np.float32)

            frames = random_sample(frames,self._num_segments) if self._is_train else fix_sample(frames,self._num_segments)
            frames = np.asarray(frames,dtype=np.float32)
            frames = self._iframe_transform(frames) if self._is_train else self._infer_transform(frames)
            frames = np.asarray(frames,dtype=np.float32) / 255.0
            frames = np.transpose(frames, (3, 0, 1, 2))
            frames = (frames - self._input_mean) / self._input_std

            pairs_data.append(frames)

        return pairs_data, self._labels_list[index]

    def __len__(self):
        return len(self._labels_list)




if __name__ == '__main__':
    import time
    start = time.time()
    train_loader = torch.utils.data.DataLoader(
        FullDecodeDataSet(
            r'/home/sjhu/datasets/all_datasets',
            video_list=r'/home/sjhu/projects/compressed_video_compare/data/datalists/debug_all_dataset.txt',
            num_segments=10,
            is_train=True
        ),
        batch_size=4, shuffle=True,
        num_workers=4, pin_memory=False)

    for i, (input_pairs, label) in enumerate(train_loader):
        iframe, mv= input_pairs
        print(iframe.shape)
        print(mv.shape)
    end = time.time()
    print("cost %f s" % ((end - start)))

