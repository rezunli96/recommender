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