"""Run training."""
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath) # 把项目的根目录添加到程序执行时的环境变量

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import argparse
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from imqfusion.dataset_gjy import CoviarDataSet
from imqfusion.model_im_fm_stack import Model
from sklearn.metrics import classification_report
import pickle
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--data-root', type=str)
parser.add_argument('--test-list', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--save-scores', type=bool, default=True)
parser.add_argument('--num-segments', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--test-crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of workers for data loader.')
parser.add_argument('--gpus', nargs='+', type=int, default=None)


LOG_ROOT_URL = r'/home/sjhu/projects/compressed_video_compare/imqfusion/'
def test(data_root,test_list,weights,num_segments,batch_size,workers,gpus):
    '''
    :param data_root:  the target video source
    :param test_list:  video pairs
    :param weights:  saved model weights path(absolute)
    :param num_segments: the TSN idea, sample frames number from video
    :param batch_size: batch size when infer
    :param workers: when using single thread,workers=0; else use mulitprocress to speed up dataloader
    :param gpus: a list of gpu device id,ex:[0,1,2,3]
    :return:
    '''
    devices = [torch.device("cuda:%d" % device) for device in gpus]
    description = 'seg_%d_%s' % (num_segments, "im_fm_stack_test")
    log_name = r'/home/sjhu/projects/compressed_video_compare/imqfusion/log/%s' % description
    writer = SummaryWriter(log_name)

    model = Model(2, num_segments)
    checkpoint = torch.load(weights)
    # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    print("model epoch {} lowest loss {}".format(checkpoint['epoch'], checkpoint['loss_min']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)


    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            data_root,
            video_list=test_list,
            num_segments=num_segments,
            is_train=False,
        ),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    model = torch.nn.DataParallel(model, device_ids=gpus)
    model = model.to(devices[0])
    cudnn.benchmark = True


    total_num = len(val_loader.dataset)
    scores = []
    labels = []
    proc_start_time = time.time()
    correct_nums=0
    for i, (input_pairs, label) in enumerate(val_loader):
        with torch.no_grad():
            input_pairs[0][0] = input_pairs[0][0].float().to(devices[0])
            input_pairs[0][1] = input_pairs[0][1].float().to(devices[0])
            input_pairs[1][0] = input_pairs[1][0].float().to(devices[0])
            input_pairs[1][1] = input_pairs[1][1].float().to(devices[0])
            label = label.float().to(devices[0])
            _, y = model(input_pairs)
            _, predicts = torch.max(y, 1)
            scores.append(y.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            correct_nums += (predicts == label.clone().long()).sum()
            cnt_time = time.time() - proc_start_time
            if (i + 1) % 100 == 0:
                print('Batch {} done, total {}/{}, average {} sec/video pair'.format(i, i + 1,
                                                                                total_num/batch_size,
                                                                    float(cnt_time) / ((i + 1)*batch_size)))

    with open(LOG_ROOT_URL + 'scores.pkl', 'wb') as fp:
        pickle.dump(scores, fp)
    fp.close()
    with open(LOG_ROOT_URL + 'labels.pkl', 'wb') as fp:
        pickle.dump(labels, fp)
    fp.close()
    acc = 100 * correct_nums / len(val_loader.dataset)
    print(('Validating Results: accuracy: {accuracy:.3f}%'.format(accuracy=acc)))
    scores = np.concatenate(scores,axis=0)
    labels = np.concatenate(labels,axis=0)
    predits = np.argmax(scores, 1).ravel()
    labels = np.around(labels).astype(np.long).ravel()
    target_names = ['Copy', 'Not Copy']
    print(classification_report(labels, predits, target_names=target_names))

def read_pkl():
    with open(LOG_ROOT_URL + 'scores.pkl', 'rb') as fp:
        scores = pickle.load(fp)
    fp.close()
    with open(LOG_ROOT_URL + 'labels.pkl', 'rb') as fp:
        labels = pickle.load(fp)
    fp.close()
    scores = np.concatenate(scores,axis=0)
    labels = np.concatenate(labels,axis=0)
    predits = np.argmax(scores, 1).ravel()
    labels = np.around(labels).astype(np.long).ravel()
    target_names = ['Copy', 'Not Copy']
    print(classification_report(labels, predits, target_names=target_names))


if __name__ == '__main__':
    pass
    # read_pkl()