import random
import numpy as np
import gc


def tail_padding(lst, least_len):
    # use last element to pad the lst
    assert len(lst) < least_len, print("no need to pad")
    padding_length = least_len - len(lst)
    tail = lst[-1]
    for i in range(padding_length):
        lst.append(tail)
    return lst


def partition(lst, n):
    if len(lst) < n:
        # print("upsample mv")
        lst = tail_padding(lst, n)
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]


def random_sample(lst, n):
    lst = np.asarray(lst)
    idxs = list(range(lst.shape[0]))
    idx_groups = partition(idxs, n)
    select_idxs = []
    for g in idx_groups:
        select_idxs.append(random.choice(g))
    mat = np.take(lst, select_idxs, axis=0)
    del idx_groups
    del select_idxs
    gc.collect()
    return mat


def fix_sample(lst, n):
    lst = np.asarray(lst)
    idxs = list(range(lst.shape[0]))
    idx_groups = partition(idxs, n)
    select_idxs = []
    for g in idx_groups:
        select_idxs.append(g[-1])
    mat = np.take(lst, select_idxs, axis=0)
    del idx_groups
    del select_idxs
    gc.collect()
    return mat


if __name__ == '__main__':
    a = list(range(100))
    b = random_sample(a, 9)
    e = fix_sample(a, 10)
    c = random_sample(a, 100)
    d = random_sample(a, 120)
    print(b)
    print(e)
    print(c)
    print(d)
    print('ok')
