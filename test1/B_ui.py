import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

f = open("train_data.pkl", 'rb')
H = pickle.load(f)
f.close()


f = open("s_ui.pkl", 'rb')
s_ui = pickle.load(f)
f.close()

f = open("s_iu.pkl", 'rb')
s_iu = pickle.load(f)
f.close()


n2 = len(H[0])
n1 = len(H)

f = open("B.pkl", "wb")
B = []
for u in range(n2):
    print(u)
    B_u = []
    for i in range(n1):
        B_ui = []
        for v in s_ui[i]:
            for j in s_iu[u]:
                if(H[j, v]): B_ui.append((v, j))
        B_u.append(B_ui)
    B.append(B_u)

pickle.dump(B, f)
f.close()
