import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

f = open("train_data.pkl", 'rb')
H = pickle.load(f)
f.close()

n2 = len(H[0])
n1 = len(H)

s_ui = []
f = open("s_ui.pkl", 'rb')
for u in range(n2):
    s_ui.append(pickle.load(f))
f.close()

s_iu = []
f = open("s_iu.pkl", 'rb')
for i in range(n1):
    s_iu.append(pickle.load(f))
f.close()




f = open("B.pkl", "wb")
for u in range(n2):
    #print(u)
    B_u = []
    for i in range(n1):
        B_ui = {}
        for v in s_ui[u][i]:
            for j in s_iu[i][u]:
                if(H[j, v]):
                    if(v in B_ui.keys()):
                        B_ui[v].append(j)
                    else:
                        B_ui[v] = [j]
        B_u.append(B_ui)
    #print(B_u)
    if(B_u == [{}] * n1):
        #print(u)
        pickle.dump(0, f)
    else: pickle.dump(B_u, f)
f.close()
