import pickle
import numpy as np
import math

n1 = 1000
n2 = 1390


f = open("sparse_train_data.pkl","rb")
y = pickle.load(f)
f.close()

f = open("sparse_test_data.pkl", 'rb')
test = pickle.load(f)
f.close()

B = []

f = open("B.pkl","rb")
for i in range(n2):
    print(i)
    B.append(pickle.load(f))
f.close()

f = open("w_vj.pkl","rb")
w = pickle.load(f)
f.close()

f = open("true_rank.pkl", "rb")
true_rank = pickle.load(f)
f.close()


ken = np.zeros(n2)
s = np.zeros(n2)
nd = np.zeros(n2)
p = np.zeros(n2)


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
        dcg += (2**(test[X[i], u]) - 1)/math.log(i+2)
        norm += (2**(test[Y[i], u]) - 1)/math.log(i+2)
    return dcg/norm


def Precision(k, u, X, Y):
    x = X[:k]
    y = Y[:k]
    count = 0
    for i in x:
        if test[i, u] == 5: count += 1
    return count/k

res = []
for u in range(n2):
    print(u)
    for i in range(n1):
        if(not y[i, u]):
            d = 0
            n = 0
            for (v, j) in B[u][i]:
                d += w[v, j] * (y[j, u] + y[i, v] - y[j, v])
                n += w[v, j]
            if(n == 0): y[i, u] = 0
            else: y[i, u] = d/n

    sigma = y[:, u]
    D = list(zip(sigma, range(len(sigma))))
    D.sort(key=lambda x: x[0], reverse=True)
    r = [x[1] for x in D if x[1] in true_rank[u]]
    r = [x[1] for x in D if x[1] in true_rank[u]]
    res.append(r)
    print(r)
    kend = kendall_tau(r, true_rank[u])
    ken[u] = kend
    print("kendall_tau for " + str(u) + " : ", kend)
    spe = spearman_rho(r, true_rank[u])
    s[u] = spe
    print("spearman_rho for " + str(u) + " : ", spe)
    ndcg = NDCG(5, u, r, true_rank[u])
    nd[u] = ndcg
    print("NDCG for " + str(u) + " : ", ndcg)
    pre = Precision(5, u, r, true_rank[u])
    p[u] = pre
    print("Precision for " + str(u) + " : ", pre)

f = open("result_la.pkl", "wb")
pickle.dump(y, f)
f.close()

f = open("res_num2.txt", 'w')

f.write("kendall_tau: "+str(float(np.mean(ken)))+"("+str(float(np.var(ken)))+")\n")
f.write("pearman_rho: "+str(float(np.mean(s)))+"("+str(float(np.var(s)))+")\n")
f.write("NDCG: "+str(float(np.mean(nd)))+"("+str(float(np.var(nd)))+")\n")
f.write("Precision: "+str(float(np.mean(p)))+"("+str(float(np.var(p)))+")\n")
