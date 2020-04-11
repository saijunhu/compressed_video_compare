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
import coviexinfo
import matplotlib.pyplot as plt
import torchvision
from utils.sample import *

from transforms import *

# from memory_profiler import profile
import gc
VIDEOS_URL = r'/home/sjhu/datasets/all_dataset'
## For Dataset
WIDTH = 256
HEIGHT = 340


class CoviarDataSet(data.Dataset):
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
        self._mv_transform = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, self._mv_scales),
             GroupRandomHorizontalFlip(is_mv=True)])
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
                video_a = video_a.split('/')[-1]
                video_b = video_b.split('/')[-1]
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
        # divide into segments, then fetch a frame in every seg
        pairs_data = []  # ( (img1,mv1), (img2,mv2) )

        for video in video_pairs:
            # shapes  (nums,height,width,channels)
            video_features = []
            extracter = VideoExtracter(video)

            # process mv
            mvs = extracter.load_mvs(self._num_segments, self._is_train)
            mvs = self._mv_transform(mvs) if self._is_train else self._infer_transform(mvs)
            mvs = np.asarray(mvs, dtype=np.float32) / 255.0
            mvs = np.transpose(mvs, (3, 0, 1, 2))
            mvs = (mvs - 0.5)

            # # process iframe
            iframes = extracter.load_keyframes(self._num_segments, self._is_train)
            iframes = self._iframe_transform(iframes) if self._is_train else self._infer_transform(iframes)
            iframes = np.asarray(iframes,dtype=np.float32) / 255.0
            iframes = np.transpose(iframes, (3, 0, 1, 2))
            iframes = (iframes - self._input_mean) / self._input_std

            video_features.append(iframes)
            video_features.append(mvs)
            pairs_data.append(video_features)

        return pairs_data, self._labels_list[index]

    def __len__(self):
        return len(self._labels_list)


class VideoExtracter:
    def __init__(self, video_name):
        # ex: filename = 916710595466737253411014029368.mp4
        os.chdir(VIDEOS_URL)
        self.video_name = video_name
        # get basic decode information
        frames_type = coviexinfo.get_num_frames(video_name)
        self.num_frames = frames_type.shape[1]
        self.num_I = np.sum(frames_type[0] == 1).item()

    def load_keyframes(self, num_segments, is_train):
        """
        :param num_segments:
        :param is_train:
        :return: (counts, width, height, channels)
        """
        os.chdir(VIDEOS_URL)
        frames = coviexinfo.extract(self.video_name, 'get_I', self.num_frames, self.num_I, 0)
        if len(frames) == 0:
            mat = np.random.randint(255, size=(num_segments, WIDTH, HEIGHT, 3))
            return np.array(mat, dtype=np.float32)

        mat = []
        for i in range(self.num_I):
            rgb = np.dstack((frames[:, :, i * 3], frames[:, :, i * 3 + 1], frames[:, :, i * 3 + 2]))
            mat.append(rgb)
            # plt.imshow(rgb)
            # plt.show()
        mat = random_sample(mat, num_segments) if is_train else fix_sample(mat, num_segments)
        mat = np.asarray(mat, dtype=np.float32)
        return mat

    # @profile
    def load_mvs(self, num_segments, is_train):
        """
        :param num_segments:
        :param is_train:
        :return: (counts, width//4, height//4, channels=2) 0,255
        """
        # mv_ref_arr=(H/4,W/4,frames*6)
        # mv_ref_arr is a array with 3 dimensions. The first dimension denotes Height of a frame. The second dimension denotes Width of a frame.
        # For every frame, it contains mv_0_x, mv_0_y, ref_0, mv_1_x, mv_1_y, ref_1. So, the third dimension denote frames*6.

        os.chdir(VIDEOS_URL)
        mv_origin = coviexinfo.extract(self.video_name, 'get_mv', self.num_frames, self.num_I, 0)
        if len(mv_origin) == 0:
            mat = np.random.randint(1, size=(num_segments, WIDTH, HEIGHT, 2))
            return np.array(mat, dtype=np.float32)

        mat = []
        mv_0_x = mv_origin[:, :, ::6]
        mv_0_y = mv_origin[:, :, 1::6]
        for i in range(mv_0_x.shape[2]):
            mv_0 = np.dstack((mv_0_x[:, :, i], mv_0_y[:, :, i]))
            mat.append(mv_0 + 128)
            # plt.imshow(mv_0)
            # plt.show()
        mat = random_sample(mat, num_segments) if is_train else fix_sample(mat, num_segments)
        mat = np.asarray(mat, dtype=np.float32)
        mv_origin = []
        return mat



def main():
    import time

    start = time.time()
    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            r'/home/sjhu/datasets/all_datasets',
            video_list=r'/home/sjhu/projects/compressed_video_compare/data/datalists/debug_all_dataset.txt',
            num_segments=10,
            is_train=True
        ),
        batch_size=4, shuffle=True,
        num_workers=4, pin_memory=False)

    for i, (input_pairs, label) in enumerate(train_loader):
        (iframe, mv), _ = input_pairs
        print(iframe.shape)
        print(mv.shape)
    end = time.time()
    print("cost %f s" % ((end - start)))

if __name__ == '__main__':
    main()
