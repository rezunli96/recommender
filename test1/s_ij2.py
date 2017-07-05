import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

f = open("train_data.pkl", 'rb')
H = pickle.load(f)
f.close()


f = open("N2ij_train.pkl", 'rb')
N_ij = pickle.load(f)
f.close()


n2 = len(H[0])
n1 = len(H)

f = open("s_ij2.pkl", "wb")
s = np.zeros((n1, n1))
for i in range(n1):
    print(i)
    for j in range(n1):
        if(len(N_ij[i][j]) < 2): continue
        if(i != j):
            entry = 0
            for u in N_ij[i][j]:
                for v in N_ij[i][j]:
                    if(u != v): entry += (H[i, u] - H[j, u] - H[i, v] + H[j, v]) ** 2
            s[i, j] = entry/(2 * len(N_ij[i][j]) * (len(N_ij[i][j]) - 1))

pickle.dump(s, f)
f.close()