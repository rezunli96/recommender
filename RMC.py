import os
import pickle
import random
import numpy as np
from sklearn.isotonic import IsotonicRegression
from metric import new_distance

dir = ".\\result"

def process_RMC(di, epsilon, tau, max_iter, K):
    d = dir + "\\" + di + "\\"  # directory to store file
    if not os.path.exists(d):
        os.makedirs(d)
    f = open(d + "train_data.pkl", "rb")

    train_data = pickle.load(f)

    f.close()

    f = open(d + "true_rank.pkl", "rb")


    true_rank = pickle.load(f)

    f.close()


    ma = -50
    mi = 50
    n1 = len(train_data)
    n2 = len(train_data[0])
    for i in range(n1):
        for j in range(n2):
            if (train_data[i, j] != -99):
                if (train_data[i, j] > ma): ma = train_data[i, j]
                if (train_data[i, j] < mi): mi = train_data[i, j]

    for i in range(n1):
        for j in range(n2):
            if (train_data[i, j] != -99):
                train_data[i, j] = (train_data[i, j] - mi) / (ma - mi)


    X = np.zeros((n1, n2))
    Z = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            if (train_data[i, j] != -99):
                X[i, j] = train_data[i, j]
                Z[i, j] = np.random.uniform()


    for i in range(max_iter):
        X_next = X + 0.5 * (Z - X)
        U, s, V = np.linalg.svd(X_next, full_matrices=False)
        for k in range(len(s)):
            s[k] = max(s[k] - tau / 2, 0)
        s = np.diag(s)
        X_next = np.dot(U, np.dot(s, V))
        for u in range(n2):
            y_u = list(zip(train_data[:, u], range(n1)))
            y_u.sort(key=lambda x: x[0])
            y_u = [x for x in y_u if x[0] != -99]
            d = np.array(range(len(y_u)))
            projected = np.array([.5 * (Z[x[1], u] + X[x[1], u]) for x in y_u]) - epsilon * d
            ir = IsotonicRegression()
            y_ = ir.fit_transform(d, projected)
            y_ += epsilon * d
            for j in range(len(y_)):
                Z[y_u[j][1], u] = y_[j]
        X = X_next

    dis = np.zeros(n2)
    for u in range(n2):
        sigma = X[:, u]
        D = list(zip(sigma, range(len(sigma))))
        D.sort(key=lambda x: x[0], reverse=True)

        res = [x[1] for x in D]
        output_rank = {}
        for i in range(len(res)):
            output_rank[res[i]] = i + 1
        dis[u] = new_distance(true_rank[u], output_rank, K)
    return dis

