"""Combine testing results of the three models to get final accuracy."""

import argparse
import numpy as np
import pickle


def combine():
    parser = argparse.ArgumentParser(description="combine predictions")
    parser.add_argument('--iframe', type=str, required=True,
                        help='iframe score file.')
    parser.add_argument('--mv', type=str, required=True,
                        help='motion vector score file.')
    parser.add_argument('--res', type=str,
                        help='residual score file.')
    parser.add_argument('--labels', type=str, required=True,
                        help='labels file')

    parser.add_argument('--wi', type=float, default=0.6,
                        help='iframe weight.')
    parser.add_argument('--wm', type=float, default=0.4,
                        help='motion vector weight.')
    parser.add_argument('--wr', type=float, default=1.0,
                        help='residual weight.')

    args = parser.parse_args()

    scores_mv = []
    scores_iframe = []
    labels = []
    with open(args.mv, 'rb') as f:
        scores_mv = pickle.load(f)

    with open(args.iframe, 'rb') as f:
        scores_iframe = pickle.load(f)

    with open(args.labels, 'rb') as f:
        labels = pickle.load(f)

    scores_mv = np.array(scores_mv)
    scores_mv = np.squeeze(scores_mv)

    scores_iframe = np.array(scores_iframe)
    scores_iframe = np.squeeze(scores_iframe)
    # scores = args.wm * scores_mv + args.wi * scores_iframe
    acc=0.0
    weight_mv = 0
    weight_iframe = 0
    for i in range(10):
        for j in range(10):
            scores = i * scores_mv + j*scores_iframe
            predits = np.argmax(scores, 1)
            labels = np.around(labels).astype(np.long).ravel()
            tmp = float((predits == labels).sum() / len(labels) * 100)
            if acc < tmp:
                weight_mv = i
                weight_iframe = j
                acc = tmp
    print("The fusion accuray is %.3f}" % acc)
    print("The weight mv is %d , weight iframe is %d}" % (weight_mv ,weight_iframe))



if __name__ == '__main__':
    combine()
