import numpy as np
import pickle
import math
import os


'''
This file store possible top-K distance function for two rank array. For example, X = [3, 4, 1, 5 .....], meaning in X

item 0 ranks the 3th highest, item 1 ranks the 4 th......

'''

def delta(j, K):
    if(j <= K): return 1
    else: return 0


def new_distance(X, Y, K):
    d = 0
    n = 0
    for i in X.keys():
        if i in Y.keys():
            n += 1
            for j in range(min(X[i], Y[i]), max(X[i], Y[i])):
                d += delta(j, K)
    if(n == 0): return 999999
    else: return d/n


def kendall_tau(x, y):
    n = 0
    d1 = 0
    d2 = 0
    for i in range(len(x)):
        for j in range(len(y)):
            a = 0
            b = 0
            if(x[j] > x[i]): a = 1
            elif(x[j] < x[i]): a = -1
            if(y[j] > y[i]): b = 1
            elif(y[j] < y[i]): b = -1
            n += a*b
            d1 += a**2
            d2 += b**2

    if(d1*d2 == 0): return 1
    return (n/(d1*d2)**.5)

def spearman_rho(x, y):
    n = len(x)
    res = 0
    for i in range(n):
        res += (x[i] - y[i]) ** 2
    res = 1 - 6 * res / (n ** 3 - n)
    return res