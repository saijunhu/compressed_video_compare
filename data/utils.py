import os
from PIL import Image
import numpy as np
import cv2

def calcul_video_mse():
    origin = np.load('residuals.npy')
    os.chdir('./jpegs')
    index=0
    mses=[]
    imgs=[]
    for img in os.listdir('/home/sjhu/projects/pytorch-coviar/data/jpegs'):
        imgs.append(img)
    imgs.sort(key= lambda x: int(x.split('.')[0]))
    for img in imgs:
        now = Image.open(img)
        now = np.array(now,dtype=np.int16)
        now = now - 128
        mses.append(rmse(origin[index],now))
        index = index+1
    mses = np.array(mses)
    print(mses.mean())
    print(mses.max())
    print(mses.min())
    return

def between_mse():
    origin = np.load('residuals.npy')

    mses=[]
    for i in range(origin.shape[0]-10):
        t=0
        for j in range(10):
            t = t + rmse(origin[j],origin[j+1])
        mses.append(t/10)

    mses = np.array(mses)
    print(mses.mean())
    print(mses.max())
    print(mses.min())
    return

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def save_as_jpeg():
    array= np.load("mvs.npy")
    min = array.min()
    max = array.max()
    from PIL import Image
    # os.mkdir('./jpegs')
    os.chdir('./jpegs')
    for i in range(array.shape[0]):
        tmp = int_to_255(array[i])
        im = Image.fromarray(tmp)
        im.save("%d.jpeg" % i)


def deduplicate(videos):
    pass
    # finished = []
    # for video in os.listdir(MVS_URL):
    #     finished.append(video+'.mp4')
    #
    # todo = []
    # for video in videos:
    #     if video not in finished:
    #         todo.append(video)
    # return todo

def free_folder_space(folder):
    return_code = os.system("rm -rf %s" % folder)
    assert return_code == 0, "Error: free folder space failed"
    # print("Free space finished.")


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

def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

def random_sample(lst,n):
    import random
    groups = partition(lst,n)
    mat = []
    for g in groups:
        mat.append(random.choice(g))
    return mat

def fix_sample(lst,n):
    import random
    groups = partition(lst,n)
    mat = []
    for g in groups:
        mat.append(g[-1])
    return mat

def convert_mv_to_3C(mat):
    # Use Hue, Saturation, Value colour model
    w = mat.shape[1], h= mat.shape[2]
    hsv = np.zeros((w, h, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(tmp[..., 0], tmp[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

if __name__ == '__main__':
    # split_datset()
    # run()
    fix_sample(list(range(100)),5)
