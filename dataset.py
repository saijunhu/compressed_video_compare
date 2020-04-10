"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors, 
or residuals) for training or testing.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data

from coviar_gjy import get_num_frames
from coviar_gjy import load
from transforms import color_aug

GOP_SIZE = 12


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


def get_gop_pos(frame_idx, representation):
    """
    :param frame_idx: the frame index in entire video
    :param representation: representation
    :return: the frame gop_index and pos_in_gop.
    """
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        # if happen I-frame,then use the pre frame to replace it.
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos


class CoviarDataSet(data.Dataset):
    def __init__(self, data_root, data_name,
                 video_list,
                 representation,
                 transform,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._data_name = data_name
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
        self._frames_list = []
        self._labels_list = []
        with open(video_list, 'r') as f:
            for line in f:
                # sample: "smile/Me_smiling_smile_h_nm_np1_fr_goo_0.avi smile 0"
                video, _, label = line.strip().split()
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                self._videos_list.append(video_path)
                self._frames_list.append(get_num_frames(video_path))
                self._labels_list.append(int(label))

        # self._videos_list = np.array(self._videos_list)
        self._frames_list = np.array(self._frames_list)
        self._labels_list = np.array(self._labels_list)

        self._size = len(self._videos_list)
        print('%d videos loaded.' % self._size)

    def _get_train_frame_index(self, num_frames, seg):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                           representation=self._representation)

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, self._representation)

    def _get_test_frame_index(self, num_frames, seg):
        # for evaluate the performance, during test, we must extract the non-random frame
        if self._representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if self._representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, self._representation)

    def __getitem__(self, index):
        # siamese label, '0' means same, '1' means diff
        siamese_label = 0

        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2
        else:
            representation_idx = 0
        index_1 = 0
        index_2 = 0
        if self._is_train:
            # 我们需要构建一个正负样本
            index_1 = random.randint(0, self._size - 1)
            index_2 = 0
            if index % 2 == 0:
                # construct a POSTIVE pair
                indices = np.squeeze(np.argwhere(self._labels_list == self._labels_list[index_1]))
                index_2 = random.choice(indices)
                siamese_label = 0
            else:
                # construct a NEG pair
                indices = np.squeeze(np.argwhere(self._labels_list != self._labels_list[index_1]))
                index_2 = random.choice(indices)
                siamese_label = 1
        else:
            # 为了检测性能，这里还是需要用均衡的数据来验证,但是要确保每次生成的对是一样的
            index_1 = index
            index_2 = 0
            if index % 2 == 0:
                # construct a POSTIVE pair
                indices = np.squeeze(np.argwhere(self._labels_list == self._labels_list[index_1]))
                random.seed(index_1)
                index_2 = random.choice(indices)
                siamese_label = 0
            else:
                # construct a NEG pair
                indices = np.squeeze(np.argwhere(self._labels_list != self._labels_list[index_1]))
                random.seed(index_1)
                index_2 = random.choice(indices)
                siamese_label = 1

        frames_all = []
        # divide into segments, then fetch a frame in every seg
        for i in (index_1, index_2):
            frames_per_sample = []
            # TODO summary: bug a few time , convert i to int
            video_path = self._videos_list[int(i)]
            for seg in range(self._num_segments):

                if self._is_train:
                    gop_index, gop_pos = self._get_train_frame_index(self._frames_list[i], seg)
                else:
                    gop_index, gop_pos = self._get_test_frame_index(self._frames_list[i], seg)

                # load(input.mp4, 3, 8, 1, True)
                # returns the accumulated motion vectors of the 9th frame of the 4th GOP.
                img = load(video_path, gop_index, gop_pos,
                           representation_idx, self._accumulate)

                if img is None:
                    print('Error: loading video %s failed.' % video_path)
                    img = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))
                else:
                    if self._representation == 'mv':
                        img = clip_and_scale(img, 20)
                        img += 128
                        img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
                    elif self._representation == 'residual':
                        img += 128
                        img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

                if self._representation == 'iframe':
                    img = color_aug(img)

                    # BGR to RGB. (PyTorch uses RGB according to doc.)
                    img = img[..., ::-1]

                frames_per_sample.append(img)
            frames_per_sample = self._transform(frames_per_sample)
            frames_per_sample = np.array(frames_per_sample)
            frames_per_sample = np.transpose(frames_per_sample, (0, 3, 1, 2))
            input = torch.from_numpy(frames_per_sample).float() / 255.0
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

            frames_all.append(input)

        return frames_all, siamese_label

    def __len__(self):
        return len(self._videos_list)



