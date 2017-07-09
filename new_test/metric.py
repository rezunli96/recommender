import numpy as np
import pickle
import math
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os

def distance1(X, Y, K):
    d = 0
    for i in range(len(X)):
        if(X[i] <= K):
            d += abs(X[i] - Y[i])
    return d