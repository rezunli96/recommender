import numpy as np
import pickle
from scipy.sparse import lil_matrix
# jit decorator tells Numba to compile this function.
f = open("train_data.pkl", "rb")
train = pickle.load(f)
f.close()



f = open("val_data.pkl", "rb")
val = pickle.load(f)
f.close()

f = open("test_data.pkl", "rb")
test = pickle.load(f)
f.close()


n1 = 1000
n2 = 1390

rate = [0] * 6

t = 0
for u in range(n2):
    total = 0
    for i in range(n1):
        rate[int(train[i, u])] += 1
        rate[int(val[i, u])] += 1
        rate[int(test[i, u])] += 1

print(rate)
'''
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
'''



