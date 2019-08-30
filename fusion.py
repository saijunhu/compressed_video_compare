import numpy as np
import torch
import pickle
from utils import metric_performance


def main(dist_1, dist_2, labels):
    distance_1 = np.array([])
    distance_2 = np.array([])
    with open(dist_1) as fp:
        distance_1 = pickle.load(dist_1)

    with open(dist_2) as fp:
        distance_2 = pickle.load(dist_1)

    with open(labels) as fp:
        labels = pickle.load(dist_1)

    distance = 0.7 * distance_1 + 0.3 * distance_2
    _, auc_score, (acc_score, recall_score, precision_score) = metric_performance(distance, labels)
    print('''The test AUC score is {:.02f},
            recall score is {:.02f}%,
            accuracy score is {:.02f}%, 
            precision score is {:.02f}%)'''.format(auc_score, recall_score, acc_score, precision_score))


if __name__ == '__main__':
    main()
