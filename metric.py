import numpy as np

def kendall_tau(X, Y):
    n = 0
    d1 = 0
    d2 = 0
    for i in range(len(X)):
        for j in range(i):
            a = 0
            b = 0
            if(X[i] > X[j]): a = 1
            elif(X[i] < X[j]): a = -1
            if(T[i] > Y[j]): b = 1
            elif(Y[i] < Y[j]): b= -1
            n += a*b
            d1 += a**2
            d2 += b**2
    return (n/(d1*d2)**.5)

def spearman_rho(X, Y):
    n = 0
    d1 = 0
    d2 = 0
    for i in range(len(X)):
        for j in range(i):
            a = X[i] - X[j]
            b = Y[i] - Y[j]
            n += a*b
            d1 += a**2
            d2 += b**2
    return (n/(d1*d2)**.5)

