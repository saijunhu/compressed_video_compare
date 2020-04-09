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
from PIL import Image
from data.coviar_gjy import VideoExtracter
import traceback
import logging

from transforms import color_aug
from transforms_hsj import transform_mv, transform_rgb_residual,transform_infer


class CoviarDataSet(data.Dataset):
    def __init__(self, data_root,
                 video_list,
                 representation,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._num_segments = num_segments
        self._representation = representation
        self._is_train = is_train
        self._accumulate = accumulate
        self._size = 0
        # modify depend on Kinetics-400 dataset setting
        self._input_mean = torch.from_numpy(
            np.array([0.43216, 0.394666, 0.37645]).reshape((3,1, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.22803, 0.22145, 0.216989]).reshape((3, 1, 1, 1))).float()

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
        one_pairs_data = []  # ( (img1,mv1), (img2,mv2) )
        try:
            keyframes_pairs = [1,2]
            # for video in video_pairs:
            #     # shapes  (nums,height,width,channels)
            #     # for rgb: all I-frame
            #     # for mv,residual: regulated by self._num_segments
            #     extracter = VideoExtracter(video)
            #     keyframes = extracter.load_keyframes(self._num_segments, self._is_train)
            #     keyframes = self.normalization(keyframes, 'iframe')
            #
            #     # assert mvs.shape[1] == self._num_segments, print("residual num_segments sample error")
            #     keyframes_pairs.append(keyframes)
            #     # video_features.append(mvs)
            #     # video_features.append(qps)


            mvs_pairs =[]
            for video in video_pairs:
                extracter = VideoExtracter(video)
                mvs = extracter.load_mvs(self._num_segments, self._is_train)
                mvs = self.normalization(mvs, 'mv')
                mvs_pairs.append(mvs)

            return [[keyframes_pairs[0],mvs_pairs[0]],[keyframes_pairs[1],mvs_pairs[1]]], self._labels_list[index]
        except Exception as e:
            traceback.print_exc()
            logging.exception(e)

    def __len__(self):
        return len(self._labels_list)

    def normalization(self,video_features, representation):
        try:
            video_features = np.array(video_features, dtype=np.uint8)
            output = []
            if self._is_train == False:
                for i in range(video_features.shape[0]):
                    t = video_features[i,...]
                    img = Image.fromarray(t)
                    output.append(transform_infer(img))
                output = torch.stack(output)
                output = np.transpose(output,(1,0,2,3)) / 255.0
                if representation == 'iframe' or representation == 'residual':
                    output = (output - self._input_mean) / self._input_std
                else:
                    output = (output - output.mean())[:2, :, :, :]
            elif representation == 'iframe' or representation == 'residual':
                for i in range(video_features.shape[0]):
                    t = video_features[i,...]
                    img = Image.fromarray(t)
                    output.append(transform_rgb_residual(img))
                output = torch.stack(output)
                output = np.transpose(output,(1,0,2,3)) / 255.0
                output = (output - self._input_mean) / self._input_std
            elif representation == 'mv':
                # torch.Size([3, 2, 224, 224])
                for i in range(video_features.shape[0]):
                    t = video_features[i,...]
                    img = Image.fromarray(t)
                    output.append(transform_mv(img))
                output = torch.stack(output)
                output = np.transpose(output,(1,0,2,3)) / 255.0
                output = (output - output.mean())[:2, :, :, :]
            else:
                assert False, print("representation wrong")
            return output
        except Exception as e:
            traceback.print_exc()
            logging.exception(e)

if __name__ == '__main__':
    import time
    start = time.time()
    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            r'/home/sjhu/datasets/all_datasets',
            video_list=r'/home/sjhu/projects/compressed_video_compare/data/datalists/all_train_sample.txt',
            num_segments=10,
            is_train=True,
            accumulate=True,
            representation='mixed'
        ),
        batch_size=1, shuffle=True,
        num_workers=8, pin_memory=False)

    for i, (input_pairs, label) in enumerate(train_loader):
        iframe = input_pairs
        # print(iframe[0].shape)
        # print(mv1.shape)
    end = time.time()
    print("cost %f s" % ((end-start)))