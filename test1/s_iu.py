import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

f = open("train_data.pkl", 'rb')
H = pickle.load(f)
f.close()

f = open("Nuv_train.pkl", 'rb')
N_uv = pickle.load(f)
f.close()

f = open("N2ij_train.pkl", 'rb')
N_ij = pickle.load(f)
f.close()


n2 = len(H[0])
n1 = len(H)

f = open("s_iu.pkl", "wb")

beta = 10

s = []

for u in range(n2):
    print(u)
    s_i = []
    for i in range(n1):
        s_iu = []
        for j in range(n1):
            if(i != j and len(N_ij[i][j]) >= beta and H[j, u]): s_iu.append(j)
        s_i.append(s_iu)
    s.append(s_i)

pickle.dump(s, f)
f.close()