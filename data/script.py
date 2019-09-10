
## FOR MEDIUM
FEATURES_URL = r'/home/sjhu/datasets/medium_dataset/features'
SPLIT_URL = r'/home/sjhu/datasets/medium_dataset/features_2'
ROOT_URL = r'/home/sjhu/datasets'
VIDEOS_URL = r'/home/sjhu/datasets/all_dataset'  #
TXT_ROOT_URL = r'/home/sjhu/datasets/medium_dataset/'

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


if __name__ == '__main__':
    # split_datset()
    # run()
    move_dataset()

