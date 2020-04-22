import os
import sys

SOURCE_URL = r'/home/sjhu/datasets/UCF-101-video-compare'
ROOT_URL = r'/home/sjhu/datasets/formal_large_dataset'
DATA_ROOT = r'/home/sjhu/datasets/formal_large_dataset/dataset'


base_cat =[]
def create(base_num):
    cnt = 0
    with open(ROOT_URL + '/base_videos.txt', 'w') as fw_base:
        with open(ROOT_URL + '/web_videos.txt', 'w') as fw_web:
            for cats in os.listdir(SOURCE_URL):
                base_cat.append(cats)
                for videos in os.listdir(os.path.join(SOURCE_URL, cats)):
                    id = videos.strip().split('/')[-1].split('_')[-1]
                    # move the origin to 200
                    origin_video = ''
                    # move the varients to 2000
                    if cnt >= base_num:
                        fw_base.close()
                        fw_web.close()
                        return
                    if id == 'c01':
                        videos_path = os.path.join(SOURCE_URL, cats, videos)
                        os.chdir(videos_path)
                        for video in os.listdir(videos_path):
                            ext = video.strip().split('.')[-1]
                            if ext == 'avi':
                                os.system('cp %s %s' % (
                                os.path.join(videos_path, video), os.path.join(ROOT_URL, 'base_video', video)))
                                origin_video = video
                                cnt = cnt + 1
                                fw_base.write(os.path.basename(video) + '\n')

                            else:  # '*.mp4'
                                os.system('cp %s %s' % (
                                os.path.join(videos_path, video), os.path.join(ROOT_URL, 'web_video', video)))
                                fw_web.write(os.path.basename(video) + '\n')
                    if id=='c02':
                        videos_path = os.path.join(SOURCE_URL, cats, videos)
                        os.chdir(videos_path)
                        for video in os.listdir(videos_path):
                            ext = video.strip().split('.')[-1]
                            if ext != 'avi':
                                os.system('cp %s %s' % (
                                    os.path.join(videos_path, video),
                                    os.path.join(ROOT_URL, 'web_video', video)))
                                fw_web.write(os.path.basename(video) + '\n')



def create_compare_pairs():
    pos_num=0
    with open(ROOT_URL + '/base_videos.txt', 'r') as fr_base:
        with open(ROOT_URL + '/web_videos.txt', 'r') as fr_web:
            lines_base = fr_base.readlines()
            lines_web = fr_web.readlines()
            with open(ROOT_URL + '/dataset.txt', 'w') as fw_dataset:
                for a in lines_base:
                    for b in lines_web:
                        if is_copy(a, b):
                            fw_dataset.write(a.strip() + ',' + b.strip() + ',' + '0\n')
                            pos_num+=1
                        else:
                            fw_dataset.write(a.strip() + ',' + b.strip() + ',' + '1\n')
            fw_dataset.close()
        fr_web.close()
    fr_base.close()
    print(pos_num)


def is_copy(base_a, web_b):
    # same video
    cat_a = base_a.strip().split('_')[1]
    group_a = base_a.strip().split('_')[2]
    cat_b = web_b.strip().split('_')[1]
    group_b = web_b.strip().split('_')[2]
    if cat_a == cat_b and group_a == group_b:
        return True
    else:
        return False


def put_all_files_in_one_directory(input, output):
    for file in os.listdir(input):
        os.system('cp %s %s' % (os.path.join(input, file), os.path.join(output, file)))

def train_test_split():
    import random
    pos_lines=[]
    neg_lines=[]
    with open(ROOT_URL + '/pos.txt', 'r') as fw_train:
        with open(ROOT_URL + '/neg.txt', 'r') as fw_test:
            pos_lines = fw_train.readlines()
            neg_lines = fw_test.readlines()
        fw_train.close()
        fw_test.close()

    random.shuffle(pos_lines)
    random.shuffle(neg_lines)
    with open(ROOT_URL + '/train.txt', 'w') as fw_train:
        a = pos_lines[:80] + neg_lines[:720]
        random.shuffle(a)
        fw_train.writelines(a)
    fw_train.close()
    with open(ROOT_URL + '/test.txt', 'w') as fw_test:
        a = pos_lines[-20:] + neg_lines[-180:]
        random.shuffle(a)
        fw_test.writelines(a)
    fw_test.close()


def pos_neg_split():
    import random
    with open(ROOT_URL + '/dataset.txt', 'r') as fr:
        with open(ROOT_URL + '/pos.txt', 'w') as fw_train:
            with open(ROOT_URL + '/neg.txt', 'w') as fw_test:
                lines = fr.readlines()
                for line in lines:
                    str = line
                    if line.strip().split(',')[-1]=='0':
                        fw_train.write(str)
                    if line.strip().split(',')[-1]=='1':
                        fw_test.write(str)
            fw_test.close()
        fw_train.close()
    fr.close()
    fr.close()

if __name__ == '__main__':
    # create(2000)
    create_compare_pairs()
    # put_all_files_in_one_directory(os.path.join(ROOT_URL, 'web_video', 'miss_video'), DATA_ROOT)
    # put_all_files_in_one_directory(os.path.join(ROOT_URL, 'web_video', 'hit_video'), DATA_ROOT)
    # put_all_files_in_one_directory(os.path.join(ROOT_URL, 'base_video'), DATA_ROOT)
    # train_test_split()
    # pos_neg_split()