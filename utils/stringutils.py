import numpy as np

def unique_chars(dataset):
    s = set()
    for line in dataset:
        for c in line:
            s.add(c)
    return list(s)

def split_xy_last_char(dataset, get_order):
    x = []
    y = []
    for sentence in dataset:
        nsentence = []
        for c in sentence[:-1]:
            nsentence.append(get_order(c))
        x.append(nsentence)
        y.append(get_order(sentence[-1]))
    return np.array(x), np.array(y)
