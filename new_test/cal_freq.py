import numpy as np
import pickle

def cal_freq(H):
    n1 = len(H)
    n2 = len(H[0])
    total = 0
    for i in range(n1):
        for j in range(n2):
            if(H[i, j] != -99): total += 1

    return total/(n1 * n2)