
import numpy as np
import pickle
from scipy.sparse import lil_matrix
# jit decorator tells Numba to compile this function.
f = open("train_data.pkl", "rb")
train = pickle.load(f)
f.close()

n1 = 1000
n2 = 1390

t = lil_matrix((n1, n2))

for i in range(n1):
    for j in range(n2):
        if(train[i, j]): t[i, j] = train[i, j]

f = open("sparse_train_data.pkl", "wb")
pickle.dump(t, f)
f.close()

f = open("test_data.pkl", "rb")
train = pickle.load(f)
f.close()


t = lil_matrix((n1, n2))

for i in range(n1):
    for j in range(n2):
        if (train[i, j]): t[i, j] = train[i, j]

f = open("sparse_test_data.pkl", "wb")
pickle.dump(t, f)
f.close()

#print(s_iu)

#print(s_ui)