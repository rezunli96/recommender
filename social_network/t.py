import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os
import matplotlib.pyplot as plt
import random


'''
x = [0.2, 0.4, 0.6, 0.8]
y1 = [1.3662, 1.07032, 0.7, 0.3]
y2 = [1.4838, 1.17032, 0.8, 0.4]
plt.plot(x, y1, marker= "*", ls = "--", label = 'New')
plt.plot(x, y2, marker= "o", ls = "-.", label = 'MRW')
plt.legend()
plt.show()
'''





x = int(np.random.poisson(2, 1))
print(x)







'''
f = open("raw_data.txt", "rb")

data = np.loadtxt(f ,delimiter=" ")

f.close()

full_data = np.zeros((1899, 1899))

for rate in data:
    full_data[int(rate[1])-1, int(rate[0])-1] = rate[2]

n = 1899
I = [0] * 1899
total = 0
for i in range(n):
    for j in range(n):
        if(full_data[i, j]): I[j] += 1

I = list(zip(I, range(len(I))))
I.sort(key=lambda x: x[0], reverse=True)
drop = [x[1] for x in I][100:]
full_data = np.delete(full_data, drop, axis=1)
full_data = np.delete(full_data, drop, axis=0)
f = open("full_data.pkl", "wb")

pickle.dump(full_data, f)

f.close()
'''