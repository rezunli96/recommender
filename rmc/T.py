import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

'''
f = open("full_data_movieLen.pkl", "rb")

full_data = pickle.load(f)

f.close()


print(full_data)

U, s, V = np.linalg.svd(full_data, full_matrices=False)

for i in range(len(s)):
    s[i] = max(s[i] - 1 / 2, 0)

print(np.dot(np.dot(U, np.diag(s)), V))
'''




def X_update(X, omega, Z, lam):

    d1 = len(X)
    d2 = len(X[0])

    X_next = np.zeros((d1, d2))

    for j in range(len(omega)):
        for i in omega[j]:
            X_next[i, j] = X[i, j] + 0.5 * (Z[i, j] - X[i, j])


    U, s, V = np.linalg.svd(X_next, full_matrices=False)
    for i in range(len(s)):
        s[i] = max(s[i] - lam/2, 0)

    return np.dot(np.dot(U, np.diag(s)), V)


def Z_update(X, Y, Z, epsilon, omega):
    d1 = len(X)
    d2 = len(X[0])
    New_Z = np.zeros((d1, d2))
    for j in range(d2):
        y_j = Y[:, j]
        y_j = list(zip(y_j, range(len(y_j))))
        y_j.sort(key=lambda x: x[0])
        rate_yj = [x[0] for x in y_j if x[0]]
        index_yj = [x[1] for x in y_j if x[0]]
        new_rate_yj = []
        d = []
        for i in range(len(rate_yj)):
            d.append(i)
        for i in range(len(rate_yj)):
            new_rate_yj = .5 * (Z[index_yj[i] ,j] + X[index_yj[i] ,j]) - epsilon * d[i]

        ir = IsotonicRegression()
        y_ = ir.fit_transform(rate_yj, new_rate_yj)
        for i in range(len(y_)):
            New_Z[index_yj[i] ,j] = y_[i]

    return New_Z












