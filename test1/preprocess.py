import numpy as np
import random
import pickle
from scipy.sparse import lil_matrix
data = np.loadtxt("ratings.dat", delimiter='::')

n1 = 3952
n2 = 6040


data = data[:, :3]

full_data = np.zeros((n1, n2))



for rate in data:
    full_data[int(rate[1]) - 1, int(rate[0]) - 1] = rate[2]



#print(full_data)

I = n1*[0]
for i in range(n1):
    for j in range(n2):
        I[i] += (full_data[i][j] > 0)


D = list(zip(I, range(len(I))))
D.sort(key=lambda x: x[0], reverse=True)

size = 1000

drop = [x[1] for x in D][size:]

print(len(drop))

full_data = np.delete(full_data, drop, axis=0)

n1 = len(full_data)
n2 = len(full_data[0])

train = np.zeros((n1, n2))
val = np.zeros((n1, n2))
test = np.zeros((n1, n2))

#print(full_data)

for i in range(len(full_data)):
    for j in range(len(full_data[0])):
        rnd = random.uniform(0, 100)
        if(rnd <=  40): train[i, j] = full_data[i, j]
        elif (rnd <= 55): val[i, j] = full_data[i, j]
        else: test[i, j] = full_data[i, j]


drop = []

for i in range(n2):
    sum_train = 0
    sum_val = 0
    sum_test = 0
    #print(i)
    for j in range(n1):
        if(train[j, i]): sum_train += 1
        if(val[j, i]): sum_val += 1
        if(test[j, i]): sum_test  += 1
    if(sum_train < 75 and (sum_val <30 or sum_test < 30)): drop.append(i)


train = np.delete(train, drop, axis=1)
val = np.delete(val, drop, axis=1)
test = np.delete(test, drop, axis=1)



f1 = open("train_data.pkl", 'wb')
f2 = open("val_data.pkl", "wb")
f3 = open("test_data.pkl", "wb")


pickle.dump(train, f1)
pickle.dump(val, f2)
pickle.dump(test, f3)

print(train.shape)
print(val.shape)
print(test.shape)
f1.close()
f2.close()
f3.close()