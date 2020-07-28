"""Run training."""
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath) # 把项目的根目录添加到程序执行时的环境变量

import argparse
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from baseline.dataset_full_decode_pytorch import FullDecodeDataSet
from baseline.model_r3d50 import BaselineModel
from sklearn.metrics import classification_report
import pickle
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--data-root', type=str)
parser.add_argument('--des', type=str)
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
args = parser.parse_args()

LOG_ROOT_URL = r'/home/sjhu/projects/compressed_video_compare/baseline/'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def main():
    print(torch.cuda.device_count())
    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))
    devices = [torch.device("cuda:%d" % device) for device in args.gpus]
    model = BaselineModel(2, args.num_segments)
    checkpoint = torch.load(args.weights,map_location={'cuda:4':'cuda:0'})
    # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    # print("model epoch {} lowest loss {}".format(checkpoint['epoch'], checkpoint['loss_min']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)


    val_loader = torch.utils.data.DataLoader(
        FullDecodeDataSet(
            args.data_root,
            video_list=args.test_list,
            num_segments=args.num_segments,
            is_train=False,
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus)
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
            if (i + 1) % 10 == 0:
                print('Batch {} done, total {}/{}, average {} sec/video pair'.format(i, i + 1,
                                                                                total_num/args.batch_size,

                                                                    float(cnt_time) / ((i + 1)*args.batch_size)))

    with open(LOG_ROOT_URL + args.des + '_scores.pkl', 'wb') as fp:
        pickle.dump(scores, fp)
    fp.close()
    with open(LOG_ROOT_URL + args.des + '_labels.pkl', 'wb') as fp:
        pickle.dump(labels, fp)
    fp.close()
    print("-----------------Infer Finished--------------")
    # acc = 100 * correct_nums / len(val_loader.dataset)
    # print(('Validating Results: accuracy: {accuracy:.3f}%'.format(accuracy=acc)))
    # scores = np.concatenate(scores,axis=0)
    # labels = np.concatenate(labels,axis=0)
    # predits = np.argmax(scores, 1).ravel()
    # labels = np.around(labels).astype(np.long).ravel()
    # target_names = ['Copy', 'Not Copy']
    # print(classification_report(labels, predits, target_names=target_names))

    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from sklearn.metrics import precision_recall_fscore_support, roc_curve,auc,accuracy_score
    import torch.nn.functional as F
    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    predits = np.argmax(scores, 1).ravel()
    labels = np.around(labels).astype(np.long).ravel()
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predits, pos_label=0, labels=[0])
    target_names = ['Copy', 'Not Copy']
    print(classification_report(labels, predits, target_names=target_names))
    prob = F.softmax(torch.from_numpy(scores), dim=1).numpy()[:, 0]
    accuracy = accuracy_score(labels,predits)
    print("accuracy is %.6f" % accuracy)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, prob, pos_label=0)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    from matplotlib.pyplot import MultipleLocator
    plt.plot([0.1,0.1],[0,1],'g--')
    plt.plot([0,1],[0.9,0.9],'g--')
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig(args.des + '_roc.png')

def read_pkl():
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from sklearn.metrics import precision_recall_fscore_support, roc_curve,auc,accuracy_score
    import torch.nn.functional as F
    with open(LOG_ROOT_URL + 'small_dataset_scores.pkl', 'rb') as fp:
        scores = pickle.load(fp)
    fp.close()
    with open(LOG_ROOT_URL + 'small_dataset_labels.pkl', 'rb') as fp:
        labels = pickle.load(fp)
    fp.close()
    scores = np.concatenate(scores,axis=0)
    labels = np.concatenate(labels,axis=0)
    predits = np.argmax(scores, 1).ravel()
    labels = np.around(labels).astype(np.long).ravel()
    accuracy = accuracy_score(labels,predits)
    print("accuracy is %.6f" % accuracy)
    # precision, recall, f1_score,_ = precision_recall_fscore_support(labels, predits, pos_label=0, labels=[0])
    # target_names = ['Copy', 'Not Copy']
    # print(classification_report(labels, predits, target_names=target_names))
    # prob = F.softmax(torch.from_numpy(scores),dim=1).numpy()[:,0]
    #
    # false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, prob,pos_label=0)
    # roc_auc=auc(false_positive_rate, true_positive_rate)
    # plt.title('ROC')
    # plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0,1],[0,1],'r--')
    # from matplotlib.pyplot import MultipleLocator
    # plt.plot([0.1,0.1],[0,1],'g--')
    # plt.plot([0,1],[0.9,0.9],'g--')
    # plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    # plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    # plt.ylabel('TPR')
    # plt.xlabel('FPR')
    # plt.savefig('small_dataset_roc.png')

if __name__ == '__main__':
     main()
    # read_pkl()
