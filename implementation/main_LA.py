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




    for num in range(4):
        opt_beta = 0
        opt_lam = 0
        opt = 99
        res = []
        for beta in range(2, int(2 * 100 * (0.2 + 0.2 * num) ** 2)):
            for lam in range(31):
                l = 0.1 * lam
                preprocess_LA(num, beta=beta)
                dis = process_LA(num, K, lam=l)
                if (np.mean(dis) < opt):
                    opt = np.mean(dis)
                    opt_beta = beta
                    opt_lam = l

                print("For prob",0.2 + 0.2 * num,"LA for ", (beta, l), "result: ", np.mean(dis))
                res.append((beta, l, np.mean(dis)))

        print("Optimal for LA: ", (opt_beta, opt_lam), opt)
        f  = open(".\\result\\"+str(num)+"\\opt_res_LA.pkl", "wb")
        pickle.dump(res, f)
        f.close()


main()