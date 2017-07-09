import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os



u = [3.3, 1.4, 9.3, 3.5, 0.1]

u = list(zip(u, range(len(u))))
u.sort(key=lambda x: x[0], reverse=True)
# print(u)
res = list(zip([x[1] for x in u if x[0]], range(1, len(u) + 1)))
res.sort(key=lambda x: x[0])
res = [x[1] for x in res]
print(res)