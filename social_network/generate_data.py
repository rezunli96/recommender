import numpy as np
import random
import pickle
import os
'''
This file generate 100 test dataset from the origin 100 * 5488 full matrix. the ith sample

stores in directory .\\result\\i

'''

dir = ".\\result"


def generate_data(num, full_data, prob):
    #print("Generating dataset for ",num)
    d = dir + "\\" + str(num) + "\\"  # directory to store file
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    sampled_data = full_data
    n1 = len(sampled_data)
    n2 = len(sampled_data[0])

    train = np.zeros((n1, n2))
    train.fill(-99)
    #val = np.zeros((n1, n2))
    #test = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            rnd =  int(np.random.poisson(sampled_data[i, j] * prob, 1))
            train[i, j] = rnd
            '''
            elif (rnd <= 55):
                val[i, j] = full_data[i, j]
            else:
                test[i, j] = full_data[i, j]
            '''

    f0 = open(d + "sampled_data.pkl", "wb")
    f1 = open(d + "train_data.pkl", 'wb')
    #f2 = open(d + "val_data.pkl", "wb")
    #f3 = open(d + "test_data.pkl", "wb")


    pickle.dump(sampled_data ,f0)
    pickle.dump(train, f1)

    print(train.shape)
    f0.close()
    f1.close()
    #print("Generating Finished.")


