from process_MR import process_MR
from preprocess_MR import preprocess_MR
from alg_new import new_alg
import numpy as np
import pickle
from generate_data import generate_data
from true_rank import cal_true

'''
Main file to run different algorithm on batch of sampled dataset
'''

test_num = 1

def main():
    f = open("full_data.pkl", "rb")
    full_data = pickle.load(f)
    f.close()

    m = 0
    opt_beta = 0
    opt_k = 0


    for i in range(test_num):
        generate_data(i, full_data)
        cal_true(i)
        for k in range(5, 51, 5):
            for beta in range(2, 21):
                preprocess_MR(i, beta)
                dis, ken = process_MR(i, "MRW", k)
                if(np.mean(ken) > m):
                    m = np.mean(ken)
                    opt_beta = beta
                    opt_k = k

                print("For", (k, beta), "kendaull's tau is", np.mean(ken))

                #print("distance metric for ", i, "is", np.mean(dis))
                #print("Kendaull's tau for ", i, "is", np.mean(ken))

    print(opt_k, opt_beta, m)
main()