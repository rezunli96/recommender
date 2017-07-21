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


def main():
    f = open("full_data.pkl", "rb")
    full_data = pickle.load(f)
    f.close()


    for num in range(4):
        generate_data(num, full_data, 0.1 + 0.1 * num)
        cal_true_rank(num)

        find_Common_Items(num)
        find_Common_Users(num)


main()