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

w = np.zeros((n2, n1))

f = open("w_vj.pkl","wb")

for v in range(n2):
    print(v)
    for j in range(n1):
        w[v, j] = math.exp(-lam * min(min(s_uv2[:, v]), min(s_ij2[:, j])))

pickle.dump(w, f)
f.close()