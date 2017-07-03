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

f = open("Nuv_candidate.pkl", 'wb')
n2 = len(H[0])
n1 = len(H)

print(n2)

beta = 10

toBeComputed = []

D = []

for u in range(n2):
    print(u)
    tmp = []
    for v in range(n2):
        if(u != v):
            if(len(N_uv[u][v]) >= beta):
                tmp.append(v)
    toBeComputed.append(tmp)


pickle.dump(toBeComputed, f)
f.close()