
import numpy as np
import pickle
from scipy.sparse import lil_matrix
# jit decorator tells Numba to compile this function.
f = open("test_data.pkl", "rb")

test = pickle.load(f)

f.close()

f = open("true_rank.pkl", "rb")

true_rank = pickle.load(f)

f.close()

for i in range(len(true_rank)):
    for j in range(len(true_rank[i])):
        print(test[true_rank[i][j], i])
f.close()