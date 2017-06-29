import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

f = open("data_ml.pkl", 'rb')
H = pickle.load(f)
f.close()

print(H)




#print(H)
n2 = 6040
n1 = 3952




def Copeland(A):
    res = 0
    I = [0]*n1
    #print(I)
    for i in range(n1):
        for j in range(n1):
            if(i != j):
                I[i] += A[i][j]

    D = list(zip(I, range(len(I))))
    D.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in D]

def N(u, v):
    res = []
    for i in range(n1):
        if(H[i, u] and H[i, v]): res.append(i)
    return res

def I(u, v):
    N_uv = N(u, v)
    res = []
    for i in range(int(len(N_uv)/2)):
        res.append((N_uv[2 * i], N_uv[2 * i + 1]))
    return res

def R(u, v):
    res = 0
    I_uv = I(u, v)
    for (s, t) in I_uv:
        res += ((H[s, u] - H[t, u]) * (H[s, v] - H[t, v]) >= 0)
    return res

def W(i, j, u, beta):
    res = []
    for v in range(n2):
        if(u != v and len(N(u, v)) >= beta and H[i, v] and H[j, v]): res.append(v)
    return res

def Pairwise_Rank(u, i, j, beta, k):
    W_iju = W(i, j, u, beta)
    W_iju.sort(key=lambda x: R(u, x), reverse= True)
    V = W_iju[0: k]
    if(V == []): return int(np.random.randint(2))
    P = [(H[i, v] > H[j, v]) - (H[i, v] < H[j, v]) for v in V]
    if(sum(P) > 0): return 1
    elif(sum(P) < 0): return 0
    else: return int(np.random.randint(2))


def Multi_Rank(beta, k):
    A = []
    e = lil_matrix((n1, n1))
    for i in range(n2):
        A.append(e)
    for u in range(n2):
        for j in range(n1):
            for i in range(j):
                if(H[i, u] and H[j, u]):
                    A[u][i, j] = (H[i, u] > H[j ,u]) + np.random.randint(2) * (H[i, u] == H[j ,u])
                    A[u][j, i] = 1 - A[u][i, j]
                else:
                    A[u][i, j] = Pairwise_Rank(u, i, j, beta, k)
                    A[u][j, i] = 1 - A[u][i, j]
    sigma = []
    #print(A)
    for A_u in A:
        A_u = A_u.toarray()
        sigma.append(Copeland(A_u))
    return sigma



print(Multi_Rank(10, 10))



