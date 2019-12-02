import os
import time
import subprocess
import random
from tqdm import tqdm
from utils import partition

base_url = r'/home/sjhu/datasets/annotation'
video_url = r'/home/sjhu/datasets/core_dataset_no_dir/'
output_url = r'/home/sjhu/datasets/all_dataset/'
root_url = r'/home/sjhu/datasets/'
txt_url = r'/home/sjhu/datasets/medium_dataset'

def put_all_files_in_one_directory(input, output):
    for dir in os.listdir(input):
        for file in os.listdir(os.path.join(input, dir)):
            os.system('cp %s %s' % (os.path.join(input, dir, file), os.path.join(output, file)))



def generate_pos_sample():
    cnt_success=0
    cnt_failed=0
    for file in os.listdir(base_url):
        with open(os.path.join(base_url, file), 'r') as fr:
            for line in tqdm(fr):
                # bug a day ,because strip()....the end_b is \n's
                video_a, video_b, start_a, end_a, start_b, end_b = line.strip().split(',')
                output_name_a = output_url + str(random.getrandbits(100)) + ".mp4"
                output_name_b = output_url + str(random.getrandbits(100)) + ".mp4"
                input_video_a = video_url + video_a
                input_video_b = video_url + video_b
                return_code_a = os.system(
                    "ffmpeg -loglevel warning -ss %s -to %s -i %s -vf scale=340:256,setsar=1:1 -c:v libx264 -coder 1 -threads 4 -cpu-used 0 %s" % (
                    start_a, end_a, input_video_a, output_name_a))
                # print("\nvideo a is ok\n")
                return_code_b = os.system(
                    "ffmpeg -loglevel warning -ss %s -to %s -i %s -vf scale=340:256,setsar=1:1 -c:v libx264 -coder 1 -threads 4 -cpu-used 0 %s" % (
                    start_b, end_b, input_video_b, output_name_b))
                # print("\nvideo a is ok\n")
                with open(os.path.join(root_url, 'all_pos_sample.txt'), 'a') as fw:
                    if return_code_a == 0 and return_code_b == 0:
                        cnt_success+=1
                        fw.write(output_name_a + ',' + output_name_b + ',' + '0\n')
                        print("success+1")
                    else:
                        os.system('rm %s' % os.path.join(output_url,output_name_a))
                        os.system('rm %s' % os.path.join(output_url,output_name_b))
                        cnt_failed+=1
                        print("failed+1")
                fw.close()
        fr.close()
        print("Total: %d pairs success, %d pairs failed" % (cnt_success,cnt_failed))


def generate_neg_sample():
    cnt_success=0
    cnt_failed=0

    files = os.listdir(base_url)
    for i in range(2000):
        file1, file2 = random.sample(files, 2)
        with open(os.path.join(base_url, file1), 'r') as fr:
            lines = fr.readlines()
            line = random.choice(lines)
            video_a, _, start_a, end_a, _, _ = line.strip().split(',')
            output_name_a = output_url + str(random.getrandbits(100)) + ".mp4"
            output_name_b = ""
            return_code_a = os.system(
                "ffmpeg -loglevel warning -ss %s -to %s -i %s -vf scale=340:256,setsar=1:1 -c:v libx264 -coder 1 -threads 4 -cpu-used 0 %s" % (
                start_a, end_a, video_url + video_a, output_name_a))
            with open(os.path.join(base_url, file2), 'r') as fr2:
                lines = fr2.readlines()
                line = random.choice(lines)
                _, video_b, _, _, start_b, end_b = line.strip().split(',')
                output_name_b = output_url + str(random.getrandbits(100)) + ".mp4"
                return_code_b = os.system(
                    "ffmpeg -loglevel warning -ss %s -to %s -i %s -vf scale=340:256,setsar=1:1 -c:v libx264 -coder 1 -threads 4 -cpu-used 0 %s" % (
                    start_b, end_b, video_url + video_b, output_name_b))
            fr2.close()
            with open(os.path.join(root_url, 'all_neg_sample.txt'), 'a') as fw:
                if return_code_a == 0 and return_code_b == 0:
                    cnt_success += 1
                    fw.write(output_name_a + ',' + output_name_b + ',' + '1\n')
                    print("success+1")
                else:
                    os.system('rm %s' % os.path.join(output_url, output_name_a))
                    os.system('rm %s' % os.path.join(output_url, output_name_b))
                    cnt_failed += 1
                    print("failed+1")
            fw.close()
        fr.close()


def train_test_spilt(test_pairs, train_pairs):
    with open(os.path.join(root_url, 'all_neg_sample.txt'), 'r') as fn:
        neg_lines = fn.readlines()
        with open(os.path.join(root_url, 'all_pos_sample.txt'), 'r') as fp:
            pos_lines = fp.readlines()
            random.shuffle(pos_lines)
            random.shuffle(neg_lines)

            test_pos_lines = pos_lines[:test_pairs//2]
            test_neg_lines =  neg_lines[:test_pairs//2]

            train_pos_lines = pos_lines[-train_pairs//2:]
            train_neg_lines = neg_lines[-train_pairs//2:]

            with open(os.path.join(txt_url, 'train_sample.txt'), 'a') as ftrain:
                for i in range(train_pairs // 2):
                    ftrain.write(train_pos_lines[i])
                    ftrain.write(train_neg_lines[i])
            ftrain.close()
            with open(os.path.join(txt_url, 'test_sample.txt'), 'a') as ftest:
                test_lines = test_pos_lines + test_neg_lines
                random.shuffle(test_lines)
                print(len(test_lines))
                ftest.writelines(test_lines)
            ftest.close()
            fp.close()
            fn.close()


def construct_debug_train_test():
    TEST_COUNT = 43
    TRAIN_COUNT = 350
    with open(os.path.join(root_url, 'neg_sample.txt'), 'r') as fn:
        neg_lines = fn.readlines()
        with open(os.path.join(root_url, 'pos_sample.txt'), 'r') as fp:
            pos_lines = fp.readlines()
            random.shuffle(pos_lines)
            random.shuffle(neg_lines)
            test_pos_lines = pos_lines[:TEST_COUNT]
            test_neg_lines = neg_lines[:TEST_COUNT]
            train_pos_lines = pos_lines[TEST_COUNT:]
            train_neg_lines = random.sample(neg_lines[TEST_COUNT:], TRAIN_COUNT)
            with open(os.path.join(root_url, 'debug_train.txt'), 'a') as ftrain:
                for i in range(TRAIN_COUNT):
                    ftrain.write(train_pos_lines[i])
                    ftrain.write(train_neg_lines[i])
            ftrain.close()
            with open(os.path.join(root_url, 'debug_test.txt'), 'a') as ftest:
                test_lines = test_pos_lines + test_neg_lines
                random.shuffle(test_lines)
                print(len(test_lines))
                ftest.writelines(test_lines)
            ftest.close()
            fp.close()
            fn.close()



def convert_videos_codec(src,dst):
    os.system('mkdir %s' % dst)
    for file in tqdm(os.listdir(src)):
        input = os.path.join(src,file)
        output = os.path.join(dst,file)
        os.system('/home/husaijun/usr/bin/ffmpeg -i %s -c:v libx264 -coder 1 -threads 4 -cpu-used 0 -vf scale=340:256,setsar=1:1 %s' % (input, output))


def generate_videos_data(folder):
    os.system('cd %s' % folder)
    for video in tqdm(os.listdir(folder)):
        os.system('extract_mvs ./+%s' % video)

def sample_dataset():
    videos = []
    with open(os.path.join(txt_url, 'test_sample.txt'), 'r') as f1:
        with open(os.path.join(txt_url,'train_sample.txt'),'r') as f2:
            for line in f1:
                video1,video2,label = line.strip().split(',')
                videos.append(video1)
                videos.append(video2)
            for line in f2:
                video1, video2, label = line.strip().split(',')
                videos.append(video1)
                videos.append(video2)
    f1.close()
    f2.close()
    with open(os.path.join(txt_url,'dataset_sample.txt'),'w') as f:
        for video in videos:
            f.write(video+'\n')
    f.close()

def temp():
    root_url = r'/home/husaijun/storage/Dataset/VCDB/entire_dataset_mvs'
    for folder in os.listdir(root_url):
        pass



if __name__ == '__main__':
    # convert_videos_codec('/home/husaijun/storage/Dataset/VCDB/one_shot_cutted_videos','/home/husaijun/storage/Dataset/VCDB/one_shot_cutted_videos_libx264')
    # generate_pos_sample()
    # generate_neg_sample()
    # # deal_neg_txt()
    train_test_spilt(340,3600)
    sample_dataset()
    # # construct_debug_train_test()
