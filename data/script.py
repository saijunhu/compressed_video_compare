
## FOR MEDIUM
FEATURES_URL = r'/home/sjhu/datasets/medium_dataset/features'
SPLIT_URL = r'/home/sjhu/datasets/medium_dataset/features_2'
ROOT_URL = r'/home/sjhu/datasets'
VIDEOS_URL = r'/home/sjhu/datasets/all_dataset'  #
TXT_ROOT_URL = r'/home/sjhu/datasets/medium_dataset/'
from utils import *
import os

def split_datset(nums=2):
    with open(os.path.join(TXT_ROOT_URL,'test_sample.txt'),'r') as f:
        lines = f.readlines()
        half = len(lines)//2
        a = lines[:half]
        b = lines[half:]
        with open(os.path.join(TXT_ROOT_URL,'test_sample_1.txt'),'w') as f1:
            f1.writelines(a)
        f1.close()
        with open(os.path.join(TXT_ROOT_URL,'test_sample_2.txt'),'w') as f2:
            f2.writelines(b)
        f2.close()
    f.close()

def move_dataset():
    videos = []
    with open(os.path.join(TXT_ROOT_URL,'test_sample_2.txt'),'r') as f1:
        a= f1.readlines()
        with open(os.path.join(TXT_ROOT_URL,'train_sample_2.txt'),'r') as f2:
            b=f2.readlines()
            c = a+b
            for line in c:
                video1,video2,label = line.strip().split(',')
                videos.append(os.path.basename(video1).split('.')[0])
                videos.append(os.path.basename(video2).split('.')[0])
    f1.close()
    f2.close()

    os.chdir(FEATURES_URL)
    for v in videos:
        os.system('cp -r %s %s' % (v,SPLIT_URL))
        print("ok")

def merge_dataset():
    os.chdir('/data/sjhu/features_2')
    for folder in os.listdir('/data/sjhu/features_2'):
        os.system('mv %s %s' % (folder, '/data/sjhu/features_1'))



def single_proecess(array):
    FEATURES_URL = r'/home/sjhu/datasets/all_dataset_iframes'
    VIDEOS_URL = r'/home/sjhu/datasets/all_dataset'  #
    for video in array:
        iframes_folder = os.path.join(FEATURES_URL, video.split('.')[0])
        os.mkdir(iframes_folder)
        os.system(
            "/home/sjhu/env/ffmpeg-4.2-amd64-static/ffmpeg -i %s -vf select='eq(pict_type\,I)' -vsync 2 -s 340x256 -f image2 %s/%%d.jpeg " % (
                os.path.join(VIDEOS_URL, video), iframes_folder))

def run():
    from multiprocessing import Process, cpu_count, Pool
    print(cpu_count())
    videos = []
    with open(os.path.join('/home/sjhu/projects/compressed_video_compare/data/datalists', 'all_dataset_sample.txt'), 'r') as f1:
        for line in f1:
            video = line.strip()
            video = os.path.basename(video)
            videos.append(video)
    f1.close()
    groups = partition(videos, cpu_count() // 4)
    p = Pool(cpu_count() // 4)
    for i in range(cpu_count() // 4):
        p.apply_async(single_proecess, (groups[i],))
    p.close()
    p.join()
    print("finished\n")

if __name__ == '__main__':
    # split_datset()
    # run()
    run()

