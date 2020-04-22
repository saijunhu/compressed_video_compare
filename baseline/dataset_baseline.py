import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath) # 把项目的根目录添加到程序执行时的环境变量

import torch.utils.data as data
from torchvision.io import read_video
import torchvision
from utils.data_utils import *
import math
from transforms_hsj import transform_mv, transform_rgb_residual,transform_infer

WIDTH = 256
HEIGHT = 340

torchvision.set_video_backend('video_reader')
class BaselineDataset(data.Dataset):
    def __init__(self, data_root,
                 video_list,
                 num_segments,
                 is_train):

        self._data_root = data_root
        self._num_segments = num_segments
        self._is_train = is_train
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
                # video_a = video_a.split('/')[-1]
                # video_b = video_b.split('/')[-1]
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
        one_pairs_data = []  # ( img1,, img2 )
        for video in video_pairs:
            # shapes  (nums,height,width,channels)
            frames = self.load_frames(video, self._num_segments, self._is_train)
            frames = self.normalization(frames, 'iframe')
            # assert frames.shape[0] == self._num_segments, print("shape wrong ")
            one_pairs_data.append(frames)

        return one_pairs_data, self._labels_list[index]

    def __len__(self):
        return len(self._labels_list)

    def normalization(self,video_features, representation):
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


    def load_frames(self, video,num_segments, is_train):
        """
        :param num_segments:
        :param is_train:
        :return: (counts, width, height, channels)
        """
        all_frames,_,_ = read_video(video) # (T, H, W, C)
        if len(all_frames):
            length = all_frames.shape[0]
        else:
            length = 0
        interval = math.ceil(length / num_segments)
        self.idxs = []
        ## for some exception
        if interval == 0:
            mat = np.random.randint(255, size=(num_segments, WIDTH, HEIGHT, 3))
            return np.array(mat, dtype=np.float32)

        if length < num_segments:
            idx = list(range(length))
        else:
            idx = list(range(length))
            if is_train:
                idx = random_sample(idx, num_segments)
                idx.sort()
            else:
                idx = fix_sample(idx, num_segments)
        mat = np.take(all_frames,idx,axis=0)

        if mat.shape[0] < num_segments:
            # use last to pad
            e = mat[-1, ...]
            e = e[np.newaxis, ...]
            pad = np.repeat(e, num_segments - mat.shape[0], axis=0)
            mat = np.concatenate((mat, pad), axis=0)

        self.idxs = idx
        return np.array(mat, dtype=np.float32).transpose((0,2,1,3))

if __name__ == '__main__':
    import time
    start = time.time()
    train_loader = torch.utils.data.DataLoader(
        BaselineDataset(
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