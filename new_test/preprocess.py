import numpy as np
import random
import pickle
import os
'''
This file generate 100 test dataset from the origin 100 * 5488 full matrix. the ith sample

stores in directory .\\result\\i


'''




dir = ".\\result"

test_num = 1 # total number of test samples

f = open("full_data.pkl", "rb") # full_data.pkl is a preprocessed 100 * 5488 full matrix from jester_1 dataset

full_data = pickle.load(f)

f.close()


def prepro(num, full_data):
    # print(num)
    d = dir + "\\" + str(num) + "\\"  # directory to store file
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    user_size = 500

    total_user_number = len(full_data[0])
    sample = random.sample(range(total_user_number), user_size) # randomly sample 300 users from 5488, range(x) = [0, 1, 2..., x]
    sampled_data = full_data[:, sample]
    sampled_data = full_data[:, :500]
    n1 = len(sampled_data)
    n2 = len(sampled_data[0])

    train = np.zeros((n1, n2))
    train.fill(-99)
    #val = np.zeros((n1, n2))
    #test = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            rnd = random.uniform(0, 100)
            if (rnd <= 40):   # split the dataset 40% into the training set
                train[i, j] = sampled_data[i, j]
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



def main():
    for i in range(test_num):
        prepro(i, full_data)


main()