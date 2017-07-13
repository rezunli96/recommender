from process_MR import process_MR
from preprocess_MR import preprocess_MR
from alg_new import new_alg
import numpy as np
import pickle
from generate_data import generate_data
from true_rank import cal_true
from preprocess_New import preprocess_new
from process_New import process_New


'''

Main file to run different algorithm on batch of sampled dataset


'''


test_num = 1

beta_new = 20

K = 10

k = 10

beta_MR = 5

def main():
    f = open("full_data.pkl", "rb")
    full_data = pickle.load(f)
    f.close()

    for i in range(test_num):
        generate_data(i, full_data)
        cal_true(i, K)
        preprocess_new(i, beta_new ,K)
        dis_new = process_New(i, K)
        print(np.mean(dis_new))
        preprocess_MR(i, beta_MR)
        dis_MR = process_MR(i, "MRW", k)
        print(np.mean(dis_MR))
main()