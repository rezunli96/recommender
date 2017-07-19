from process_MR import process_MR
from preprocess_MR import preprocess_MR
from alg_new import new_alg
import numpy as np
import pickle
from generate_data import generate_data
from true_rank import cal_true
from preprocess_New import preprocess_new
from process_New import process_New
from N_uv import N

'''

Main file to run different algorithm on batch of sampled dataset


'''


test_num = 1



#beta_new = 5

# K = 10


#k_new = 20

#k_MR = 20

#beta_MR = 5

def main():
    f = open("full_data.pkl", "rb")
    full_data = pickle.load(f)
    f.close()

    new_to_K = []

    mr_to_K = []

    for i in range(test_num):
        generate_data(i, full_data, 0.3)
        N(i)
        K = 100
        cal_true(i, K)
        preprocess_new(i, 10, 25, K)
        dis_new = process_New(i, K)
        print("For new alg K = ", K, "distance is", np.mean(dis_new))
        new_to_K.append((K, np.mean(dis_new)))
        preprocess_MR(i, 10)
        dis_MR = process_MR(i, "MRW", 25, K)
        print("For MRW K = ", K, "distance is", np.mean(dis_MR))
        mr_to_K.append((K, np.mean(dis_MR)))
        '''
        for j in range(10, 100, 20):
            K = j
            cal_true(i, K)
            preprocess_new(i, 10, 25, K)
            dis_new = process_New(i, K)
            print("For new alg K = ", K, "distance is", np.mean(dis_new))
            new_to_K.append((K, np.mean(dis_new)))
            preprocess_MR(i, 10)
            dis_MR = process_MR(i, "MRW", 25, K)
            print("For MRW K = ", K, "distance is", np.mean(dis_MR))
            mr_to_K.append((K, np.mean(dis_MR)))
        '''
    '''
    f = open("new_to_K.pkl", "wb")
    pickle.dump(new_to_K, f)
    f.close()

    f = open("mr_to_K.pkl", "wb")
    pickle.dump(mr_to_K, f)
    f.close()
    '''



main()