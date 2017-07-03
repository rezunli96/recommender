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

f = open("s_ui.pkl", "wb")

beta = 10

s = []

for i in range(n1):
    print(i)
    s_u = []
    for u in range(n2):
        s_ui = []
        for v in range(n2):
            if(u != v and len(N_uv[u][v]) >= beta and H[i, v]): s_ui.append(v)
        s_u.append(s_ui)
    s.append(s_u)

pickle.dump(s, f)
f.close()

