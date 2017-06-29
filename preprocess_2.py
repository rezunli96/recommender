import numpy as np
import os
from scipy import sparse
import pickle
path = "F:\\lz\\cornell\\lab\\dataset\\download\\training_set\\"
d = {}

current = 0
num_u = 480189
num_i = 17770

res = sparse.dok_matrix((num_i,num_u),dtype=np.float32)

output = open("data_nf.pkl",'wb')
for filename in os.listdir(path):
    f = np.loadtxt(path+filename, usecols=(0, 1),
                   delimiter=',', skiprows=1)
    ind = int(filename[3:10])
    #print(ind)
    for rate in f:
        if rate[0] not in d.keys():
            d[rate[0]] = current
            current+=1
        res[ind - 1,d[rate[0]]] = rate[1]
    #print(res)
    #print(int(filename[3:10]))
   # print(current)
#print(current)
pickle.dump(res,output)
output.close()


