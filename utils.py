## This file contains Loss, Metrics

import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_auc_score
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

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


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class WarmStartCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    add warm start to CosineAnnealingLR
    """
    def __init__(self, optimizer, T_max, T_warm, eta_min=0, last_epoch=-1,):
        self.T_max = T_max
        self.T_warm = T_warm
        self.eta_min = eta_min

        super(WarmStartCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warm:
            return [base_lr * ((self.last_epoch+1) / self.T_warm) for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


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


def visualize(distnces, labels, title=""):
    plt.scatter(distnces, labels)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Distance')
    plt.ylabel('Truth label')
    plt.title(title)
    plt.show()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]


def smooth_one_hot(true_labels: torch.Tensor, num_classes: int, smoothing=0.0):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    otherwise = smoothing / (num_classes - 1)
    with torch.no_grad():
        true_dist = torch.zeros((true_labels.shape[0], num_classes), dtype=torch.float32, device=true_labels.device)
        true_dist.fill_(otherwise)
        true_dist.scatter_(1, true_labels.long().unsqueeze(1), confidence)
    return true_dist

def smoothed_label_loss(pred,smoothed_label):
    log_prob = torch.nn.functional.log_softmax(pred,dim=1)
    return -torch.sum(log_prob*smoothed_label)/pred.shape[0]

if __name__ == '__main__':
    pass
