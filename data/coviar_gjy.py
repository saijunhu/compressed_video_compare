import coviexinfo
import matplotlib.pyplot as plt
from utils.sample import *
import math
import os
import numpy as np
import cv2
import traceback

VIDEOS_URL = r'/home/sjhu/datasets/all_dataset'
## For Dataset
WIDTH = 256
HEIGHT = 340


class VideoExtracter:
    def __init__(self, video_name):
        # ex: filename = 916710595466737253411014029368.mp4
        os.chdir(VIDEOS_URL)
        self.video_name = video_name
        ## get basic decode information
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
        frames = coviexinfo.extract(self.video_name, 'get_I', self.num_frames, self.num_I, 1)
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

    def load_mvs(self, num_segments, is_train):
        """
        :param num_segments:
        :param is_train:
        :return: (counts, width//4, height//4, channels=3) 0,255
        """
        # mv_ref_arr=(H/4,W/4,frames*6)
        # mv_ref_arr is a array with 3 dimensions. The first dimension denotes Height of a frame. The second dimension denotes Width of a frame.
        # For every frame, it contains mv_0_x, mv_0_y, ref_0, mv_1_x, mv_1_y, ref_1. So, the third dimension denote frames*6.
        try:
            os.chdir(VIDEOS_URL)
            mv_origin = coviexinfo.extract(self.video_name, 'get_mv', self.num_frames, self.num_I, 0)
            if len(mv_origin) == 0:
                mat = np.random.randint(1, size=(num_segments, WIDTH, HEIGHT, 3))
                return np.array(mat, dtype=np.float32)

            mat = []
            mv_frames = mv_origin.shape[2]//6
            print(mv_frames)
            for i in range(mv_frames):
                mat.append(np.dstack((mv_origin[:, :, i * 6] + 128, mv_origin[:, :, i * 6 + 1] + 128, np.zeros((WIDTH // 4, HEIGHT // 4), dtype=np.int32))))
                # mat.append(mv_0)
                # plt.imshow(mv_0)
                # plt.show()
            mat = random_sample(mat, num_segments) if is_train else fix_sample(mat, num_segments)
            mat = np.asarray(mat, dtype=np.float32)
            return mat
        except Exception as e:
            print("When extracting MV, something wrong")
            traceback.print_exc()


if __name__ == '__main__':
    extracter = VideoExtracter("290218917876718905070219193338.mp4")
    # extracter.load_mvs(5, True)
    extracter.load_keyframes(5,True)

    # def load_residuals(self, num_segments, is_train):
    #     """
    #     :param num_segments:
    #     :param is_train:
    #     :return: (counts, width, height, channels), 0,255
    #     """
    #     os.chdir(VIDEOS_URL)
    #     Y_origin = extract(self.video_name, 2, self.num_frames)
    #     UV_origin = extract(self.video_name, 3, self.num_frames)
    #     if len(Y_origin) == 0:
    #         mat = np.random.randint(1, size=(num_segments, WIDTH, HEIGHT, 3))
    #         return np.array(mat, dtype=np.float32)
    #     Y = Y_origin.transpose((2, 0, 1))
    #     U = []
    #     V = []

    #     for i in range(UV_origin.shape[2]):
    #         if i % 2 == 0:
    #             u = UV_origin[..., i]
    #             v = UV_origin[..., i + 1]
    #             extend_u = cv2.resize(u.astype(np.uint8), dsize=(HEIGHT, WIDTH), interpolation=cv2.INTER_CUBIC)
    #             extend_v = cv2.resize(v.astype(np.uint8), dsize=(HEIGHT, WIDTH), interpolation=cv2.INTER_CUBIC)
    #             U.append(extend_u)
    #             V.append(extend_v)
    #
    #     dst = np.stack((Y, np.array(U, dtype=np.int32), np.array(V, dtype=np.int32)), axis=3)
    #     residuals = []
    #     for i in range(dst.shape[0]):
    #         rgb = cv2.cvtColor(dst[i, ...].astype(np.float32), cv2.COLOR_YUV2RGB)
    #         residuals.append(rgb + 128)
    #     residuals = np.array(residuals)
    #
    #     mat = []
    #     length = self.num_frames
    #     interval = math.ceil(length / num_segments)
    #
    #     ## for some except
    #     if interval == 0:
    #         mat = np.random.randint(1, size=(num_segments, WIDTH, HEIGHT, 3))
    #         return np.array(mat, dtype=np.float32)
    #
    #     if length < num_segments:
    #         idx = list(range(length))
    #     else:
    #         idx = list(range(length))
    #         if is_train:
    #             idx = random_sample(idx, num_segments)
    #             idx.sort()
    #         else:
    #             idx = fix_sample(idx, num_segments)
    #
    #     mat = np.take(residuals, idx, axis=0)
    #     # mat = mat - 128
    #     if mat.shape[0] < num_segments:
    #         # use last to pad
    #         e = mat[-1, ...]
    #         e = e[np.newaxis, ...]
    #         pad = np.repeat(e, num_segments - mat.shape[0], axis=0)
    #         mat = np.concatenate((mat, pad), axis=0)
    #     return np.array(mat, dtype=np.float32)

    # def load_qp(self, num_segments):
    #     # return (count, 1, WIDTH/16,HEIGHT/16)
    #     os.chdir(VIDEOS_URL)
    #     QP_SIZE = 56
    #     depth_qp_arr = extract(self.video_name, 1, self.num_frames)
    #     if len(depth_qp_arr) == 0:
    #         return np.full((num_segments, 1, QP_SIZE, QP_SIZE), 0.5, dtype=np.float32)
    #     depth = []
    #     QP = []
    #     for i in range(depth_qp_arr.shape[2]):
    #         if i % 2 == 0:
    #             depth.append(depth_qp_arr[..., i])
    #             QP.append(depth_qp_arr[..., i + 1])
    #     qp = np.array(QP)
    #     # TODO update idx
    #     idx = self.idxs
    #     result = []
    #     if qp.shape[0] < len(idx) or len(idx) == 0:
    #         return np.full((num_segments, 1, QP_SIZE, QP_SIZE), 0.5, dtype=np.float32)
    #
    #     for i in idx:
    #         result.append(qp[i])
    #     assert len(result) != 0, print(" result shape wrong ")
    #     result = np.array(result, dtype=np.float32)
    #     assert len(idx) <= num_segments, print("idx len greater than num_segments")
    #     if len(idx) < num_segments:
    #         mat = np.full((result.shape[1], result.shape[2]), 26.0)
    #         mat = mat[np.newaxis, ...]
    #         mat = np.repeat(mat, num_segments - result.shape[0], axis=0)
    #         result = np.concatenate((result, mat), axis=0)
    #     outputs = []
    #     for i in range(result.shape[0]):
    #         outputs.append(cv2.resize(result[i], dsize=(QP_SIZE, QP_SIZE), interpolation=cv2.INTER_CUBIC))
    #     outputs = 1 - (np.array(outputs, dtype=np.float32) / 51)
    #     return np.expand_dims(outputs, axis=1)


