import pickle
import numpy as np

f = open("train_data.pkl","rb")
y = pickle.load(f)
f.close()

f = open("B.pkl","rb")
B = pickle.load(f)
f.close()

f = open("w_vj.pkl","rb")
w = pickle.load(f)
f.close()

n1 = len(y)
n2 = len(y[0])

for i in range(n1):
    for u in range(n2):
        if(not y[i, u]):
            d = 0
            n = 0
            for (v, j) in B[u][i]:
                d += w[v, j] * (y[u, j] + y[v, i] - y[v, j])
                n += w[v, j]
            y[i, u] = d/n

f = open("result_la.pkl", "wb")
pickle.dump(y, f)
f.close()