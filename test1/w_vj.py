import numpy as np
import random
import pickle
import math
f = open("train_data.pkl", 'rb')
H = pickle.load(f)
f.close()

f = open("s_ij2.pkl", "rb")
s_ij2 = pickle.load(f)
f.close()

f = open("s_uv2.pkl", "rb")
s_uv2 = pickle.load(f)
f.close()


n1 = len(H)
n2 = len(H[0])

lam = 2

w = []

f = open("w_vj.pkl","wb")

for v in range(n2):
    w_v = []
    print(v)
    for j in range(n1):
        w_vj = []
        for u in range(n2):
            w_vju = []
            for i in range(n1):
                w_vju.append(math.exp(-lam * min(s_uv2[u][v], s_ij2[i][j])))
            w_vj.append(w_vju)
        w_v.append(w_vj)
    w.append(w_v)

pickle.dump(w, f)
f.close()