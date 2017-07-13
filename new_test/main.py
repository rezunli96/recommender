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

K = 10


#k_new = 20

#k_MR = 20

#beta_MR = 5

def main():
    f = open("full_data.pkl", "rb")
    full_data = pickle.load(f)
    f.close()

    opt_k_new = 0
    opt_beta_new = 0
    opt_k_MR = 0
    opt_beta_MR = 0

    d_new = 999
    d_MR = 999
    for i in range(test_num):
        generate_data(i, full_data)
        cal_true(i, 10)
        N(i)



    result_new = []
    result_MR = []


    for k in range(5, 51, 5):
        for beta in range(2, 21):
            average_dis_new = 0
            average_dis_MR = 0
            for i in range(test_num):
                preprocess_new(i, beta ,k, K)
                dis_new = process_New(i, K)
                print("For subsample",i,"distance of new alg is: ",np.mean(dis_new))
                print("\n")
                average_dis_new += np.mean(dis_new)
                preprocess_MR(i, beta)
                dis_MR = process_MR(i, "MRW", k, K)
                print("For subsample", i, "distance of new MRW is: ", np.mean(dis_MR))
                average_dis_MR += np.mean(dis_MR)
                print("\n")

            average_dis_new/=test_num
            average_dis_MR/=test_num
            result_new.append((k, beta, average_dis_new))
            result_MR.append((k, beta, average_dis_MR))
            if(average_dis_new < d_new):
                opt_k_new = k
                opt_beta_new = beta
                d_new = average_dis_new

            if (average_dis_MR < d_MR):
                opt_k_MR = k
                opt_beta_MR = beta
                d_MR = average_dis_MR
            print("For ",(k, beta), "new algorithm distance is", average_dis_new)
            print("\n")
            print("For ", (k, beta), "MRW distance is", average_dis_MR)
            print("\n")

    f = open("result_new.pkl", "wb")
    pickle.dump(result_new, f)
    f.close()
    f = open("result_MR.pkl", "wb")
    pickle.dump(result_MR, f)
    f.close()
    print("The optimal parameter for new algorithm is", (opt_k_new, opt_beta_new), "with minmized distance", d_new)
    print("\n")
    print("The optimal parameter for MRW algorithm is", (opt_k_MR, opt_beta_MR), "with minmized distance", d_MR)
    print("\n")
main()