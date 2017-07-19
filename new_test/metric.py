import numpy as np
import pickle
import math
import os


'''
This file store possible top-K distance function for two rank array. For example, X = [3, 4, 1, 5 .....], meaning in X

item 0 ranks the 3th highest, item 1 ranks the 4 th......

'''



def distance1(X, Y, K):
    d = 0
    for i in range(len(X)):
        if(X[i] <= K):
            d += abs(X[i] - Y[i])
    return d




def distance2(X, Y):
    n = 0
    d = 0
    for i in X.keys():
        if i in Y.keys():
            n += 1
            d += abs(X[i] - Y[i])

    if(n == 0): return 9999999
    return d/n


def distance2_with_weight(X, Y, K):
    n = 0
    d = 0
    for i in X.keys():
        if i in Y.keys():
            n += 1
            if(X[i] == K + 1): continue
            d += (1/(X[i])) * abs(X[i] - Y[i])

    if(n == 0): return 9999999
    return d/n

def distance2_with_rate(X, Y, H):
    n = 0
    d = 0
    for i in X.keys():
        if i in Y.keys():
            n += 1
            d += (H[i]) * abs(X[i] - Y[i])

    if(n == 0): return 9999999
    return d/n



def KEN(x, y):
    n = len(x)
    pair = 0
    for j in range(n):
        for i in range(j):
            if((x[i] < x[j] and y[i] > y[j]) or (x[i] > x[j] and y[i] < y[j])):
                pair += 1
    return (2 * pair/(n * (n - 1 )))





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