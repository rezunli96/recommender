import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

f = open("data_ml.pkl", 'rb')
H = pickle.load(f)
f.close()

true_rank = []
for user in H:
    D = list(zip(user, range(len(user))))
    D.sort(key=lambda x: x[0], reverse=True)
    res = [x[1] for x in D]
    true_rank.append(res)
print(np.array(true_rank))