import numpy as np
import os
import random
import cv2
from functools import cmp_to_key
from PIL import Image
from tqdm import tqdm

## FOR SMALL DATASET
# MVS_URL = r'/home/sjhu/datasets/small_dataset/samples_mvs'
# KEYFRAMES_URL = r'/home/sjhu/datasets/small_dataset/samples_keyframes'
# FEATURES_URL = r'/home/sjhu/datasets/small_dataset/samples_features'
# ROOT_URL = r'/home/sjhu/datasets'
# VIDEOS_URL = r'/home/sjhu/datasets/all_dataset'
# TXT_ROOT_URL = r'/home/sjhu/datasets/small_dataset/'

## FOR MEDIUM
MVS_URL = r'/home/sjhu/datasets/medium_dataset/samples_mvs'
KEYFRAMES_URL = r'/home/sjhu/datasets/medium_dataset/samples_keyframes'
FEATURES_URL = r'/home/sjhu/datasets/medium_dataset/samples_features'
ROOT_URL = r'/home/sjhu/datasets'
VIDEOS_URL = r'/home/sjhu/datasets/all_dataset'
TXT_ROOT_URL = r'/home/sjhu/datasets/medium_dataset/'


class VideoExtracter:
    def __init__(self, video_name):
        # ex: filename = 916710595466737253411014029368.mp4
        self.video_name = video_name
        self.video_folder = video_name.split('.')[0]
        self.mvs_folder = MVS_URL
        self.keyframes_folder = KEYFRAMES_URL
        self.features_folder = FEATURES_URL
        self.y_files = []
        self.u_files = []
        self.v_files = []
        self.depth_files = []
        self.qp_files = []
        self.mv_x_files = []
        self.mv_y_files = []
        self.pic_names = []

    def save_video_level_features(self):
        # for mvs and residuals
        filenames = []
        for file in os.listdir(os.path.join(self.mvs_folder, self.video_folder)):
            filename = os.path.join(self.mvs_folder, self.video_folder, file)
            filenames.append(filename)
        filenames.sort(key=self.sort_by_frame_order)
        self.extract_files_by_type(filenames)
        self.residuals = self.deal_residual_matrixs()
        self.mvs = self.deal_mv_matrix()

        # for keyframes

        for pic in os.listdir(os.path.join(self.keyframes_folder, self.video_folder)):
            picname = os.path.join(self.keyframes_folder, self.video_folder, pic)
            self.pic_names.append(picname)
        self.pic_names.sort(key=self.sort_by_image_order)
        # TODO make sure the sort order
        self.keyframes = self.deal_keyframes_matrix()
        # TODO check the numpy shapes

        # save all *.npy file
        os.mkdir(os.path.join(self.features_folder, self.video_folder))
        np.save(os.path.join(self.features_folder, self.video_folder, 'residuals.npy'), self.residuals)
        np.save(os.path.join(self.features_folder, self.video_folder, 'mvs.npy'), self.mvs)
        np.save(os.path.join(self.features_folder, self.video_folder, 'keyframes.npy'), self.keyframes)
        print("Hint: features have been saved.")
        print("residuals shape:")
        print(self.residuals.shape)
        print("mvs shape:")
        print(self.mvs.shape)
        print("keyframes shape:")
        print(self.keyframes.shape)

    def load_video_level_features(self):
        residuals= self.load_residuals()
        mvs= self.load_mvs()
        keyframes= self.load_keyframes()
        return residuals, mvs, keyframes

    def load_keyframes(self):
        os.chdir(os.path.join(self.features_folder, self.video_folder))
        keyframes = np.load('keyframes.npy')
        return keyframes

    def load_mvs(self):
        os.chdir(os.path.join(self.features_folder, self.video_folder))
        mvs = np.load('mvs.npy')
        return mvs

    def load_residuals(self):
        os.chdir(os.path.join(self.features_folder, self.video_folder))
        residuals = np.load('residuals.npy')
        return residuals

    def get_num_frames(self):
        return len(self.y_files)

    def extract_files_by_type(self, filenames):
        self.u_files = [file for file in filenames if 'D_U' in file]
        self.y_files = [file for file in filenames if 'D_Y' in file]
        self.v_files = [file for file in filenames if 'D_V' in file]
        self.depth_files = [file for file in filenames if 'depth' in file]
        self.qp_files = [file for file in filenames if 'QP' in file]
        # for B-frame , here just abondon the back refernce frame
        self.mv_x_files = [file for file in filenames if '_mv_0_x' in file]
        self.mv_y_files = [file for file in filenames if 'mv_0_y' in file]

    def sort_by_frame_order(self, elem):
        return int(elem.split('/')[-1].split('_')[0])

    def sort_by_image_order(self,elem):
        return int(elem.split('/')[-1].split('.')[0])

    def deal_residual_matrixs(self):
        video_level_residuel = []
        for i in range(len(self.u_files)):
            U = np.loadtxt(self.u_files[i])
            V = np.loadtxt(self.v_files[i])
            Y = np.loadtxt(self.y_files[i])
            row, col = Y.shape
            ######  key code ######
            extend_u = cv2.resize(U, dsize=(col, row), interpolation=cv2.INTER_CUBIC)
            extend_v = cv2.resize(V, dsize=(col, row), interpolation=cv2.INTER_CUBIC)
            dst = cv2.merge((Y, extend_v, extend_u))
            rgb = cv2.cvtColor(dst.astype(np.float32), cv2.COLOR_YUV2RGB)
            video_level_residuel.append(rgb)
        # np.save(os.path.join(root, video, 'residuals.npy'), np.array(video_level_residuel, dtype=np.float64))
        # print("Hint: residual matrix has been saved.")
        return np.array(video_level_residuel, dtype=np.int8)

    def deal_residual_matrix_reduced(self):
        Ys = []
        Us = []
        Vs = []
        for i in tqdm(range(len(self.u_files))):
            U = np.loadtxt(self.u_files[i])
            V = np.loadtxt(self.v_files[i])
            Y = np.loadtxt(self.y_files[i])
            Us.append(U)
            Vs.append(V)
            Ys.append(Y)
        # np.save(os.path.join(root, 'Ys.npy'), np.array(Ys, dtype=np.float64))
        # np.save(os.path.join(root, 'Us.npy'), np.array(Us, dtype=np.float64))
        # np.save(os.path.join(root, 'Vs.npy'), np.array(Vs, dtype=np.float64))
        print("Hint: residual matrix has been saved.")
        return np.array(Ys, dtype=np.int8)

    def deal_mv_matrix(self):
        video_level_mv = []
        for i in range(len(self.mv_x_files)):
            mv_x = np.loadtxt(self.mv_x_files[i])
            mv_y = np.loadtxt(self.mv_y_files[i])
            row, col = mv_x.shape
            ######  key code ######
            extend_mv_x = cv2.resize(mv_x, dsize=(col * 4, row * 4), interpolation=cv2.INTER_CUBIC)
            extend_mv_y = cv2.resize(mv_y, dsize=(col * 4, row * 4), interpolation=cv2.INTER_CUBIC)
            mvs = cv2.merge((extend_mv_x, extend_mv_y))
            video_level_mv.append(mvs)
        # np.save(os.path.join(root, video, 'mv.npy'), np.array(video_level_mv, dtype=np.float64))
        # print("Hint: mv matrix has been saved.")
        return np.array(video_level_mv, dtype=np.int8)

    def deal_mv_matrix_reduced(self):
        video_level_mv = []
        for i in tqdm(range(len(self.mv_x_files))):
            mv_x = np.loadtxt(self.mv_x_files[i])
            mv_y = np.loadtxt(self.mv_y_files[i])
            mvs = cv2.merge((mv_x, mv_y))
            video_level_mv.append(mvs)
        # np.save(os.path.join(root, 'mv_raw.npy'), np.array(video_level_mv, dtype=np.float64))
        # print("Hint: mv matrix has been saved.")
        return np.array(video_level_mv, dtype=np.int8)

    def deal_keyframes_matrix(self):
        video_level_keyframes = []
        for picfile in self.pic_names:
            img = np.array(Image.open(picfile))
            video_level_keyframes.append(img)
        return np.array(video_level_keyframes, dtype=np.int8)


def free_folder_space(folder):
    return_code = os.system("rm -rf %s" % folder)
    assert return_code == 0, "Error: free folder space failed"
    print("Free space finished.")


def visualize_residual(filename, videoname):
    print("start io read...")
    mat = np.load(filename)
    mat = mat.astype(np.uint8)
    print("io finished")
    from cv2 import VideoWriter, VideoWriter_fourcc
    n, w, h, c = mat.shape
    FPS = 20
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # bug here
    video = VideoWriter(videoname, fourcc, FPS, (h, w))
    for i in tqdm(range(n)):
        tmp = mat[i, :, :, :]
        frame = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()


def visualize_mv(filename, videoname):
    print("start io read...")
    mat = np.load(filename)
    # mat = mat.astype(np.uint8)
    print("io finished")
    from cv2 import VideoWriter, VideoWriter_fourcc
    n, w, h, c = mat.shape
    FPS = 20
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = VideoWriter(videoname, fourcc, FPS, (h, w))
    for i in tqdm(range(n)):
        tmp = mat[i, :, :, :]
        # Use Hue, Saturation, Value colour model
        hsv = np.zeros((w, h, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(tmp[..., 0], tmp[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        video.write(bgr_frame)
    video.release()


def extract_keyframes(videos_folder=VIDEOS_URL):
    videos= []
    with open(os.path.join(TXT_ROOT_URL, 'dataset_sample.txt'), 'r') as f1:
        for line in f1:
            video = line.strip().split('/')[-1]
            videos.append(video)
    f1.close()

    for video in videos:
        output_folder = os.path.join(KEYFRAMES_URL,
                                     video.split('.')[0])

        os.system('mkdir %s' % output_folder)
        os.system("ffmpeg -i %s -vf select='eq(pict_type\,I)' -vsync 2 -s 340x256 -f image2 %s/%%d.jpeg " % (
            os.path.join(videos_folder, video), output_folder))


def extract_mvs(videos_folder=VIDEOS_URL):
    # os.system('cd %s' % videos_folder) // this command can't change working dir
    os.chdir(videos_folder)
    videos= []
    with open(os.path.join(TXT_ROOT_URL, 'dataset_sample.txt'), 'r') as f1:
        for line in f1:
            video = line.strip()
            videos.append(video)
    f1.close()
    print(len(videos))
    return videos
    # for video in tqdm(videos):
    #     video = os.path.basename(video)
    #     os.system('~/env/extract_mvs %s' % video)
    #     src_folder = os.path.join(videos_folder, video.split('.')[0])
    #     dst_folder = os.path.join(MVS_URL)
    #     # os.system('mkdir %s' % dst_folder)
    #     os.system('mv %s %s' % (src_folder, dst_folder))


def consumetime(videos):
    videos_folder = VIDEOS_URL
    for video in tqdm(videos):
        video = os.path.basename(video)
        os.system('~/env/extract_mvs %s' % video)
        src_folder = os.path.join(videos_folder, video.split('.')[0])
        dst_folder = os.path.join(MVS_URL)
        # os.system('mkdir %s' % dst_folder)
        os.system('mv %s %s' % (src_folder, dst_folder))


def feature_extracter(videos_folder=VIDEOS_URL):
    videos= []
    with open(os.path.join(TXT_ROOT_URL, 'dataset_sample.txt'), 'r') as f1:
        for line in f1:
            video = line.strip()
            videos.append(video)
    f1.close()
    return videos

    #split to 3 process to acclerate
    # for video in tqdm(videos[10:]):
    #     video = os.path.basename(video)
    #     extracter = VideoExtracter(video)
    #     extracter.save_video_level_features()

def consumetime_fe(videos):
    for video in tqdm(videos):
        video = os.path.basename(video)
        extracter = VideoExtracter(video)
        extracter.save_video_level_features()



def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

def deduplicate(videos):
    finished = []
    for video in os.listdir(MVS_URL):
        finished.append(video+'.mp4')

    todo = []
    for video in videos:
        if video not in finished:
            todo.append(video)
    return todo

if __name__ == '__main__':
    # extract_keyframes()
    # print("STEP 1 finished\n")
    from multiprocessing import Process,cpu_count,Pool
    print(cpu_count())
    todo = extract_mvs()
    groups = partition(todo,cpu_count()//4)
    p = Pool(cpu_count()//4)
    for i in range(cpu_count()//4):
        p.apply_async(consumetime, (groups[i],))
    p.close()
    p.join()
    print("STEP 2 finished\n")
    todo = feature_extracter()
    groups = partition(todo,cpu_count()//4)
    p = Pool(cpu_count()//4)
    for i in range(cpu_count()//4):
        p.apply_async(consumetime_fe,(groups[i],))
    p.close()
    p.join()
    print("STEP 3 finished\n")
        #test()
    # extract_keyframes()
    # extract_mvs()
    # feature_extracter()
