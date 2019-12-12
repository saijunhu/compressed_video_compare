import coviexinfo
import matplotlib.pyplot as plt
import numpy as np
import os
'''
extract('BasketballDrill.avi',representation,num_frames)
representation==0 denotes extracting mvs and ref.
representation==1 denotes extracting depth and qp.
representation==2 denotes extracting residual Y.
representation==3 denotes extracting residual U and V.
'''

import numpy as np
import os
from datetime import datetime
import random
import cv2
from functools import cmp_to_key
from PIL import Image
from tqdm import tqdm
from data.utils import *
from utils import *
import math
from config import *

## FOR SMALL DATASET
# MVS_URL = r'/home/sjhu/datasets/small_dataset/samples_mvs'
# KEYFRAMES_URL = r'/home/sjhu/datasets/small_dataset/samples_keyframes'
# FEATURES_URL = r'/home/sjhu/datasets/small_dataset/samples_features'
# ROOT_URL = r'/home/sjhu/datasets'
# VIDEOS_URL = r'/home/sjhu/datasets/all_dataset'
# TXT_ROOT_URL = r'/home/sjhu/datasets/small_dataset/'

## FOR MEDIUM
FEATURES_URL = r'/home/sjhu/datasets/medium_dataset/features'
ROOT_URL = r'/home/sjhu/datasets/medium_dataset/datasets'
VIDEOS_URL = r'/home/sjhu/datasets/all_dataset'  #
TXT_ROOT_URL = r'/home/sjhu/datasets/medium_dataset/'

## For Dataset
WIDTH = 256
HEIGHT = 340


class VideoExtracter:
    def __init__(self, video_name):
        # ex: filename = 916710595466737253411014029368.mp4
        os.chdir(VIDEOS_URL)
        self.video_name = video_name
        self.num_frames = coviexinfo.get_num_frames(self.video_name)
        self.pic_names = []


        # for keyframes
        # os.system(
        #     "/home/sjhu/env/ffmpeg-4.2-amd64-static/ffmpeg -i %s -vf select='eq(pict_type\,I)' -vsync 2 -s 340x256 -f image2 %s/%%d.jpeg " % (
        #         os.path.join(VIDEOS_URL, self.video_name), self.keyframes_folder))

    #TODO keyframes
    def load_keyframes(self, num_segments, is_train):
        """
        :param num_segments:
        :param is_train:
        :return: (counts, width, height, channels)
        """
        os.chdir(VIDEOS_URL)
        mat = []
        files = os.listdir(self.keyframes_folder)
        length = len(files)
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
        for i in idx:
            img = files[i]
            temp = np.asarray(Image.open(img))
            mat.append(temp)
        mat = np.array(mat, dtype=np.int16)
        if mat.shape[0] < num_segments:
            # use last to pad
            e = mat[-1, ...]
            e = e[np.newaxis, ...]
            pad = np.repeat(e, num_segments - mat.shape[0], axis=0)
            mat = np.concatenate((mat, pad), axis=0)

        self.idxs = idx
        return np.array(mat, dtype=np.float32)

    def load_mvs(self, num_segments, is_train):
        """
        :param num_segments:
        :param is_train:
        :return: (counts, width, height, channels=3) 0,255
        """
        # mv_ref_arr=(H/4,W/4,frames*6)
        # mv_ref_arr is a array with 3 dimensions. The first dimension denotes Height of a frame. The second dimension denotes Width of a frame.
        # For every frame, it contains mv_0_x, mv_0_y, ref_0, mv_1_x, mv_1_y, ref_1. So, the third dimension denote frames*6.

        os.chdir(VIDEOS_URL)
        mv_origin = coviexinfo.extract(self.video_name, 0, self.num_frames)
        mvs = []
        for i in list(range(mv_origin.shape[2])):
            if i%6==0:
                mv_x = mv_origin[:,:,i]
                extend_mv_x = cv2.resize(np.array(mv_x+128,dtype=np.uint8),dsize=(HEIGHT,WIDTH),interpolation=cv2.INTER_CUBIC)
                mv_y = mv_origin[:,:,i+1]
                extend_mv_y = cv2.resize(np.array(mv_y+128,dtype=np.uint8),dsize=(HEIGHT,WIDTH),interpolation=cv2.INTER_CUBIC)
                mv_zero = np.zeros((WIDTH,HEIGHT),dtype=np.uint8)
                mvs.append(np.dstack((extend_mv_x,extend_mv_y,mv_zero)))
        mv = np.array(mvs,dtype=np.uint8)
        mat = []
        length = self.num_frames
        interval = math.ceil(length / num_segments)

        ## for some except
        if interval == 0:
            mat = np.random.randint(1, size=(num_segments, WIDTH, HEIGHT, 3))
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
        for i in idx:
            img = mv[i,...]
            mat.append(img)
        mat = np.array(mat, dtype=np.int16)
        # mat = mat - 128
        if mat.shape[0] < num_segments:
            # use zero to pad
            pad = np.zeros((num_segments - mat.shape[0], WIDTH, HEIGHT, 3))
            mat = np.concatenate((mat, pad), axis=0)
        # return np.array(mat[..., :2], dtype=np.float32)
        return np.array(mat, dtype=np.float32)

    def load_residuals(self, num_segments, is_train):
        """
        :param num_segments:
        :param is_train:
        :return: (counts, width, height, channels), 0,255
        """
        os.chdir(VIDEOS_URL)
        Y_origin = coviexinfo.extract(self.video_name, 2, self.num_frames)
        UV_origin = coviexinfo.extract(self.video_name, 3, self.num_frames)
        Y =  Y_origin.transpose((2,0,1))
        U = []
        V=[]
        for i in range(UV_origin.shape[2]):
            if i%2==0:
                u = UV_origin[...,i]
                v = UV_origin[...,i+1]
                extend_u = cv2.resize(u.astype(np.uint8), dsize=(HEIGHT, WIDTH), interpolation=cv2.INTER_CUBIC)
                extend_v = cv2.resize(v.astype(np.uint8), dsize=(HEIGHT, WIDTH), interpolation=cv2.INTER_CUBIC)
                U.append(extend_u)
                V.append(extend_v)

        dst = np.stack((Y,np.array(U,dtype=np.int32),np.array(V,dtype=np.int32)),axis=3)
        residuals= []
        for i in range(dst.shape[0]):
            rgb = cv2.cvtColor(dst[i,...].astype(np.float32), cv2.COLOR_YUV2RGB)
            residuals.append(rgb+128)
        residuals = np.array(residuals)

        mat = []
        length = self.num_frames
        interval = math.ceil(length / num_segments)

        ## for some except
        if interval == 0:
            mat = np.random.randint(1, size=(num_segments, WIDTH, HEIGHT, 3))
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
        for i in idx:
            temp = residuals[i,...]
            mat.append(temp)
        mat = np.array(mat, dtype=np.int16)
        # mat = mat - 128
        if mat.shape[0] < num_segments:
            # use last to pad
            e = mat[-1, ...]
            e = e[np.newaxis, ...]
            pad = np.repeat(e, num_segments - mat.shape[0], axis=0)
            mat = np.concatenate((mat, pad), axis=0)
        return np.array(mat, dtype=np.float32)

    def load_qp(self, num_segments):
        # return (count, 1, WIDTH/16,HEIGHT/16)
        os.chdir(VIDEOS_URL)
        QP_SIZE = 56
        depth_qp_arr = coviexinfo.extract(self.video_name, 1, self.num_frames)
        depth=[]
        QP = []
        for i in range(depth_qp_arr.shape[2]):
            if i%2==0:
                depth.append(depth_qp_arr[...,i])
                QP.append(depth_qp_arr[...,i+1])
        qp = np.array(QP)
        #TODO update idx
        idx =self.idxs
        result = []
        if qp.shape[0] < len(idx) or len(idx) == 0:
            return np.full((num_segments, 1, QP_SIZE, QP_SIZE), 0.5,dtype=np.float32)

        for i in idx:
            result.append(qp[i])
        assert len(result)!=0, print(" result shape wrong ")
        result = np.array(result, dtype=np.float32)
        assert len(idx) <= num_segments, print("idx len greater than num_segments")
        if len(idx) < num_segments:
            mat = np.full((result.shape[1], result.shape[2]), 26.0)
            mat = mat[np.newaxis, ...]
            mat = np.repeat(mat, num_segments - result.shape[0], axis=0)
            result = np.concatenate((result, mat), axis=0)
        outputs = []
        for i in range(result.shape[0]):
            outputs.append(cv2.resize(result[i], dsize=(QP_SIZE, QP_SIZE), interpolation=cv2.INTER_CUBIC))
        outputs = 1 - (np.array(outputs, dtype=np.float32) / 51)
        return np.expand_dims(outputs, axis=1)



if __name__ == '__main__':
    extracter = VideoExtracter("999671226656870541070085860219.mp4")
    mv = extracter.load_qp(10)


#example 1, extracting mvs and ref.
os.chdir('/home/sjhu/datasets')
video = 'test.mp4'
num_frames=coviexinfo.get_num_frames('test.mp4')
#mv_ref_arr=(H/4,W/4,frames*6)
#mv_ref_arr is a array with 3 dimensions. The first dimension denotes Height of a frame. The second dimension denotes Width of a frame.
#For every frame, it contains mv_0_x, mv_0_y, ref_0, mv_1_x, mv_1_y, ref_1. So, the third dimension denote frames*6.
mv_ref_arr=coviexinfo.extract(video,0,num_frames)
f=mv_ref_arr[:,:,::6]
print(f.shape)
print(mv_ref_arr.shape)

#example 2, extracting depth and qp.
#depth_qp_arr=(H/16,W/16,frames*2),for every frame, it contains depth and qp.
depth_qp_arr=coviexinfo.extract(video,1,num_frames)
print(depth_qp_arr.shape)

#example 3, extracting residual Y.
#res_Y_arr=(H,W,frames)
res_Y_arr=coviexinfo.extract(video,2,num_frames)
print(res_Y_arr.shape)

#examples 4, extracting residual U and V.
#res_UV_arr=(H/2,W/2,frames*2), for every frame, it contains residual U and residual V.
res_UV_arr=coviexinfo.extract(video,3,num_frames)
print(res_UV_arr.shape)