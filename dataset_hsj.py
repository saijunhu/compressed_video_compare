"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors,
or residuals) for training or testing.
"""

import os
import os.path
import random
import math
import numpy as np
import torch
import torch.utils.data as data

# from coviar import get_num_frames
# from coviar import load
from data.construct_data import VideoExtracter

from transforms import color_aug


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def get_seg_range(num_frames, num_segments, seg, representation):
    if representation in ['residual', 'mv']:
        num_frames -= 1

    seg_size = float(num_frames - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg + 1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        # Exclude the 0-th frame, because it's an I-frmae.
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end


class CoviarDataSet(data.Dataset):
    def __init__(self, data_root,
                 video_list,
                 representation,
                 transform,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._num_segments = num_segments
        self._representation = representation
        self._transform = transform
        self._is_train = is_train
        self._accumulate = accumulate
        self._size = 0
        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

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

        # self._videos_list = np.array(self._videos_list)
        # self._frames_list = np.array(self._frames_list)
        self._labels_list = np.array(self._labels_list)

        self._size = len(self._labels_list)
        print('%d pair videos loaded.' % self._size)

    def __getitem__(self, index):
        # siamese label, '0' means same, '1' means diff
        video_pairs = self._videos_list[index]
        # divide into segments, then fetch a frame in every seg
        one_pairs_data = []
        for video in video_pairs:
            # shapes  (nums,height,width,channels)
            # for rgb: all I-frame
            # for mv,residual: regulated by self._num_segments
            extracter = VideoExtracter(video)
            video_features = []
            if self._representation == 'iframe':
                video_features = extracter.load_keyframes()
            elif self._representation == 'residual':
                video_features = extracter.load_residuals(self._num_segments)
                assert video_features.shape[0] == self._num_segments, print("num_segments sample error")

            elif self._representation == 'mv':
                video_features = extracter.load_mvs(self._num_segments)
                assert video_features.shape[0] == self._num_segments, print("num_segments sample error")

            video_features = np.array(video_features, dtype=np.float32)
            video_features = self._transform(video_features)
            video_features = np.array(video_features)
            video_features = np.transpose(video_features, (0, 3, 1, 2))
            input = torch.from_numpy(video_features).float() / 255.0
            # when mv, the num_segments=3
            # The input shape is:
            # torch.Size([3, 2, 224, 224])
            # The input label is:  37
            if self._representation == 'iframe':
                input = (input - self._input_mean) / self._input_std
                # was a bug here, input_mean,std is a tensor ,so we must convert input a tensor
            elif self._representation == 'residual':
                input = (input - 0.5) / self._input_std
            elif self._representation == 'mv':
                input = (input - 0.5)

            one_pairs_data.append(input)

        return one_pairs_data, self._labels_list[index]

    def __len__(self):
        return len(self._labels_list)
