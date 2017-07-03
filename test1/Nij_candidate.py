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

f = open("Nij_candidate.pkl", 'wb')
n2 = len(H[0])
n1 = len(H)

print(n2)

beta = 10

toBeComputed = []

D = []

for i in range(n1):
    print(i)
    tmp = []
    for j in range(n1):
        if(i != j):
            if(len(N_ij[i][j]) >= beta):
                tmp.append(j)
    toBeComputed.append(tmp)


pickle.dump(toBeComputed, f)
f.close()