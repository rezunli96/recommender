import numpy as np
import math
import pickle
from itertools import permutations

f = open("test_data.pkl", "rb")
H = pickle.load(f)
f.close()


def kendall_tau(X, Y):
    D = {}
    for i in range(len(Y)):
        D[Y[i]] = i + 1
    x = [D[i] for i in X]
    y = list(range(1, len(Y) + 1))
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
            elif(y[j] < y[i]): b= -1
            n += a*b
            d1 += a**2
            d2 += b**2

    if(d1*d2 == 0): return 1
    return (n/(d1*d2)**.5)

def spearman_rho(X, Y):
    D = {}
    for i in range(len(Y)):
        D[Y[i]] = i + 1
    x = [D[i] for i in X]
    y = list(range(1, len(Y) + 1))
    n = 0
    d1 = 0
    d2 = 0
    for i in range(len(x)):
        for j in range(len(y)):
            a = x[j] - x[i]
            b = y[j] - y[i]
            n += a*b
            d1 += a**2
            d2 += b**2

    if(d1*d2 == 0): return 1
    return (n/(d1*d2)**.5)


def NDCG(k, u , X, Y):
    dcg = 0
    norm = 0
    for i in range(k):
        dcg += (2**(H[X[i], u]) - 1)/math.log(i+1)
        norm += (2**(H[Y[i], u]) - 1)/math.log(i+1)
    return dcg/norm


def Precision(k, X, Y):
    x = X[:k]
    y = Y[:k]
    count = 0
    for i in x:
        if i in y: count += 1
    return count/k


X = [1, 2 ,4 ,9 ,3]

Y = [5, 1, 9, 4 ,3]


print(NDCG(list(reversed(X)), X))
