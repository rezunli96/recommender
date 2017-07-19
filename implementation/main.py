from process_MR import process_MR
from preprocess_MR import preprocess_MR
import numpy as np
import pickle
from generate_data import generate_data
from cal_true_rank import cal_true_rank
from preprocess_New import preprocess_new
from process_New_average_agg import process_New_average_agg
from process_New_medium_agg import process_New_medium_agg
from find_Common_Items import find_Common_Items
from find_Common_Users import find_Common_Users
from preprocess_LA import preprocess_LA
from process_LA import process_LA
'''

Main file to run different algorithm on batch of sampled dataset


'''
test_num = 1

#beta_new = 5

K = 10

#k_new = 20

#k_MR = 20

#beta_MR = 5

def main():
    f = open("full_data.pkl", "rb")
    full_data = pickle.load(f)
    f.close()

    for num in range(test_num):
        generate_data(num, full_data, 0.4)
        cal_true_rank(num)
        find_Common_Items(num)
        find_Common_Users(num)
        preprocess_new(num, 15, 30, K)
        dis_new = process_New_medium_agg(num, K)
        print(np.mean(dis_new))
        '''
        preprocess_LA(num, 2)
        dis_LA = process_LA(num, K, 2)
        print(np.mean(dis_LA))
        preprocess_MR(num, 2)
        dis_MR = process_MR(num, "MRW", 5, 10)
        print(np.mean(dis_MR))
        '''
main()