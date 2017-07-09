import numpy as np
import random
import pickle
import os

import numpy as np
import pickle
import math
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os
from metric import distance1


dir = ".\\result"

test_num = 100

T = 20


K = 30

def find_N(num):
    d = dir + "\\" + str(num) + "\\"
    print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "complete_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()
    n2 = len(data[0])
    n1 = len(data)
    f = open(d + "dis_uv.pkl", 'rb')
    dis_uv = pickle.load(f)
    f.close()

    N = []
    for u in range(n2):
        N_u = []
        for v in range(n2):
            if(dis_uv[u][v] < T):
                N_u.append(v)
        N.append(N_u)
    f = open(d + "N_u.pkl", "wb")
    pickle.dump(N, f)
    f.close()


for i in range(test_num):
    find_N(i)