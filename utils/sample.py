
def tail_padding(lst,least_len):
    # use last element to pad the lst
    assert len(lst) < least_len,print("no need to pad")
    padding_length = least_len - len(lst)
    tail = lst[-1]
    for i in range(padding_length):
        lst.append(tail)
    return lst



def partition(lst, n):
    if len(lst) < n:
        # print("upsample mv")
        lst = tail_padding(lst,n)
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

def random_sample(lst,n):
    import random
    groups = partition(lst,n)
    mat = []
    for g in groups:
        mat.append(random.choice(g))
    return mat

def fix_sample(lst,n):
    import random
    groups = partition(lst,n)
    mat = []
    for g in groups:
        mat.append(g[-1])
    return mat

if __name__ == '__main__':
    a = list(range(100))
    b = random_sample(a,9)
    c = random_sample(a,100)
    d = random_sample(a,120)
    print('ok')