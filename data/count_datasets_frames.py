
import coviexinfo
import os
import sys
from utils.utils import AverageMeter
from tqdm import tqdm
import random
def count(data_root):
    os.chdir(data_root)
    average_frames = AverageMeter()
    videos = os.listdir(data_root)
    random.shuffle(videos)
    cnt=0
    try:
        for video in tqdm(videos):
            frames_type = coviexinfo.get_num_frames(video)
            average_frames.update(frames_type.shape[1])
            cnt+=1
            if cnt%1000==0:
                print("This dataset average frames number is %f" % average_frames.avg)
    except:
        print("This dataset average frames number is %f" % average_frames.avg)
    else:
        print("This dataset average frames number is %f" % average_frames.avg)

if __name__ == '__main__':
    data_root =r'/home/sjhu/datasets/formal_large_dataset/dataset'
    count(data_root)