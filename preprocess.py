import numpy as np
import pickle
data = np.loadtxt("ratings.dat", delimiter='::')

n2 = 6040
n1 = 3952




data = data[:, :3]

print(data)
res = np.zeros((n1,n2))
for rate in data:
    res[int(rate[1])-1][int(rate[0])-1] = rate[2]

output = open("data_ml.pkl",'wb')


'''
for i in range(n1):
    for j in range(n2-1):
        output.write(str(res[i][j])+',')
    output.write((str(res[i][n2-1]))+'\n')
'''

pickle.dump(res,output)
output.close()
