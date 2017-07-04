import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import math
f = open("train_data.pkl", 'rb')
H = pickle.load(f)
f.close()


f = open("test_data.pkl", 'rb')
test = pickle.load(f)
f.close()

f = open("Ruv_train.pkl", 'rb')
R = pickle.load(f)
f.close()

f = open("Nuv_candidate.pkl", 'rb')
N_C = pickle.load(f)
f.close()


f = open("true_rank.pkl", "rb")
true_rank = pickle.load(f)
f.close()

print(H)

#print(H)
n2 = len(H[0])
n1 = len(H)

beta = 10

ken = np.zeros(n2)
s = np.zeros(n2)
nd = np.zeros(n2)
p = np.zeros(n2)

ken_pkl = open("ken_pkl.pkl", "wb")
s_pkl = open("s_pkl.pkl", "wb")
nd_pkl = open("nd_pkl.pkl", "wb")
p_pkl = open("p_pkl.pkl", "wb")

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







def W(i, j, u):
    res = []
    for v in N_C[u]:
        if(H[i, v] and H[j, v]): res.append(v)
    return res



def Pairwise_Rank(u, i, j, k):
    #print(((u, i, j)))
    W_iju = W(i, j, u)
    W_iju.sort(key=lambda x: R[u][x], reverse= True)
    V = W_iju[0: k]
    if(V == []): return int(np.random.randint(2))
    P = [R[u][v] * (H[i, v] > H[j, v]) - (H[i, v] < H[j, v]) for v in V]
    if(sum(P) > 0): return 1
    elif(sum(P) < 0): return 0
    else: return int(np.random.randint(2))




def Multi_Rank(k):
    res = []
    #sigma = np.zeros((n2, n1))
    for u in range(n2):
        print(u)
        sigma = [0] * n2
        for j in range(n1):
            for i in range(j):
                if(H[i, u] and H[j, u]):
                    sigma[i] += (H[i, u] > H[j ,u]) + np.random.randint(2) * (H[i, u] == H[j ,u])
                    sigma[j] += 1 - sigma[i]
                else:
                    #print((u, i, j))
                    sigma[i] += Pairwise_Rank(u, i, j, k)
                    sigma[j] += 1 - sigma[i]

        D = list(zip(sigma, range(len(sigma))))
        D.sort(key=lambda x: x[0], reverse=True)

        r = [x[1] for x in D if x[1] in true_rank[u]]
        res.append(r)
        print(r)
        kend = kendall_tau(r, true_rank[u])
        ken[u] = kend
        print("kendall_tau for "+str(u)+" : ", kend)
        spe = spearman_rho(r, true_rank[u])
        s[u] = spe
        print("spearman_rho for "+str(u)+" : ", spe)
        ndcg = NDCG(5, u, r, true_rank[u])
        nd[u] = ndcg
        print("NDCG for "+str(u)+" : ", ndcg)
        pre = Precision(5, u, r, true_rank[u])
        p[u] = pre
        print("Precision for "+str(u)+" : ", pre)
    return res







pickle.dump(ken, ken_pkl)
pickle.dump(s, s_pkl)
pickle.dump(nd, nd_pkl)
pickle.dump(p, p_pkl)

ken_pkl.close()
s_pkl.close()
nd_pkl.close()
p_pkl.close()


f = open("result1.pkl", "wb")
res = Multi_Rank(10)
pickle.dump(res, f)
f.close()

f = open("res_num_MR.txt", 'w')

f.write("kendall_tau: "+str(float(np.mean(ken)))+"("+str(float(np.var(ken)))+")\n")
f.write("pearman_rho: "+str(float(np.mean(s)))+"("+str(float(np.var(s)))+")\n")
f.write("NDCG: "+str(float(np.mean(nd)))+"("+str(float(np.var(nd)))+")\n")
f.write("Precision: "+str(float(np.mean(p)))+"("+str(float(np.var(p)))+")\n")

f.close()


