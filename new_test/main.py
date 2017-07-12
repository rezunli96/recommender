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
    for i in range(test_num):
        generate_data(i, full_data)
        cal_true(i)


    for i in range(test_num):
        preprocess_MR(i)


    for i in range(test_num):
        dis, ken = process_MR(i, "MRW")
        print("distance metric for ", i, "is", np.mean(dis))
        print("Kendaull's tau for ", i, "is", np.mean(ken))



main()