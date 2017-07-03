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

f = open("s_ui.pkl", 'rb')
s_ui = pickle.load(f)
f.close()

f = open("s_iu.pkl", 'rb')
s_iu = pickle.load(f)
f.close()


n2 = len(H[0])
n1 = len(H)

f = open("s_uv2.pkl", "wb")
s = np.zeros((n2, n2))
for u in range(n2):
    print(u)
    for v in range(n2):
        if(len(N_uv[u][v]) < 2): continue
        if(u != v):
            entry = 0
            for i in N_uv[u][v]:
                for j in N_uv[u][v]:
                    if(i != j): entry += (H[i, u] - H[i, v] - H[j, u] + H[j, v]) ** 2
            s[u, v] = entry/(2 * len(N_uv[u][v]) * (len(N_uv[u][v]) - 1))

pickle.dump(s, f)
f.close()

