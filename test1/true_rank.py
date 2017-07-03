import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

f = open("test_data.pkl", 'rb')
H = pickle.load(f)
f.close()


n1 = len(H)
n2 = len(H[0])

true_rank = []

#f = open("true_rank.txt", "w")

for i in range(n2):
    u = H[:, i]
    u = list(zip(u, range(len(u))))
    u.sort(key=lambda x: x[0], reverse=True)
    #print(u)
    res = [x[1] for x in u if x[0] and x[0] == 5]
    true_rank.append(res)
    print(len(res), len(u), len(res)/len(u))
    #f.write(",".join([str(x) for x in res])+'\n')

#f.close()
print(n2)
'''
print(true_rank)
f = open("true_rank.pkl", "wb")
pickle.dump(true_rank, f)
f.close()
'''
