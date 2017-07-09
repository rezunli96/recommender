import numpy as np
import random
import pickle
import os
from scipy.sparse import lil_matrix
#data = np.loadtxt("ratings.dat", delimiter='::')




dir = ".\\result"

alg_name = ["MR", "MRW", "MR_realvote"]

test_num = 100

f = open("complete_data.pkl", "rb")

full_data = pickle.load(f)

f.close()




def prepro(num, full_data):
    print(num)
    d = dir + "\\" + str(num) + "\\"
    print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    user_size = 300

    n1 = len(full_data)
    n2 = len(full_data[0])
    drop = random.sample(range(len(full_data[0])), len(full_data[0]) - user_size)
    full_data = np.delete(full_data, drop, axis=1)
    # print(D[:2000])

    '''
    drop = []
    for i in range(n2):
        if(I[i] < 5): drop.append(i)
    full_data = np.delete(full_data, drop, axis= 1)

    print(full_data.shape)
    '''

    n1 = len(full_data)
    n2 = len(full_data[0])

    train = np.zeros((n1, n2))
    #val = np.zeros((n1, n2))
    #test = np.zeros((n1, n2))

    # print(full_data)

    for i in range(len(full_data)):
        for j in range(len(full_data[0])):
            rnd = random.uniform(0, 100)
            if (rnd <= 40):
                train[i, j] = full_data[i, j]
            '''
            elif (rnd <= 55):
                val[i, j] = full_data[i, j]
            else:
                test[i, j] = full_data[i, j]
            '''

    drop = []
    '''
    for i in range(n2):
        sum_train = 0
        sum_val = 0
        sum_test = 0
        # print(i)
        for j in range(n1):
            if (train[j, i]): sum_train += 1
            if (val[j, i]): sum_val += 1
            if (test[j, i]): sum_test += 1
        # if(sum_train < 50 and (sum_val <25 or sum_test < 25)): drop.append(i)
        if (sum_train < 5 and sum_test < 2): drop.append(i)

    train = np.delete(train, drop, axis=1)
    val = np.delete(val, drop, axis=1)
    test = np.delete(test, drop, axis=1)
    '''
    f0 = open(d + "complete_data.pkl", "wb")
    f1 = open(d + "train_data.pkl", 'wb')
    #f2 = open(d + "val_data.pkl", "wb")
    #f3 = open(d + "test_data.pkl", "wb")

    pickle.dump(train, f1)
    pickle.dump(full_data,f0)
    #pickle.dump(val, f2)
    #pickle.dump(test, f3)

    print(train.shape)
    #print(val.shape)
    #print(test.shape)
    f0.close()
    f1.close()
    #f2.close()
    #f3.close()


for i in range(test_num):
    prepro(i, full_data)







