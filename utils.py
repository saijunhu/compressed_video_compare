## This file contains Loss, Metrics

import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_auc_score
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
import pickle


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # print("The pair label is:")
        # print(label)
        # print("The loss is %f" % loss_contrastive)
        return loss_contrastive


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def similarity(output_1, output_2):
    return F.pairwise_distance(output_1, output_2)

def accauacy(predicts, labels):
    auc_score = roc_auc_score(labels, predicts)
    acc_score = accuracy_score(labels, predicts)
    return acc_score

def metric_performance(distances, labels):
    """
    :param distances:
    :param labels:
    :return:  a tuple (best predict labels, best auc score,other_metric)
    """
    threshold = np.arange(2, 5, 0.5)
    predicts = []

    for ts in threshold:
        pred = np.copy(distances)
        pred[pred >= ts] = 1
        pred[pred < ts] = 0
        predicts.append(pred)

    plot_x = []
    plot_y = []
    best_score = -1.0
    other_metrics = ()
    for i in range(len(threshold)):
        recal_score = recall_score(labels, predicts[i], pos_label=0)
        acc_score = accuracy_score(labels, predicts[i])
        prec_score = precision_score(labels, predicts[i], pos_label=0)
        auc_score = roc_auc_score(labels, predicts[i])
        if best_score < auc_score:
            predict_best_idx = i
            best_score = auc_score
            other_metrics = (acc_score, recal_score, prec_score)
        plot_x.append(i)
        plot_y.append(auc_score)

    plt.plot(plot_x, plot_y)
    plt.ylabel("auc socore")
    plt.xlabel("threshold")
    plt.show()
    return predicts[predict_best_idx], best_score, other_metrics


def metric_performance_offline():
    import pickle
    global distances
    global labels
    with open('distance', 'rb') as f:
        distances = pickle.load(f)
    with open('truth_labels') as f:
        labels = pickle.load(f)
    return metric_performance(distances, labels)


def visualize(distnces, labels,title=""):
    plt.scatter(distnces, labels)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Distance')
    plt.ylabel('Truth label')
    plt.title(title)
    plt.show()




if __name__ == '__main__':
    pass
