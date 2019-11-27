import numpy as np


def boolarr2int(arr):
    string = '0b'
    for i in arr:
        string += str(int(i))
    return eval(string)


a = np.asarray([True, False, True, True])
print(boolarr2int(a))