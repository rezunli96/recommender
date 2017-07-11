from process_MR import process_MR
from alg_new import new_alg
import numpy as np


'''
Main file to run different algorithm on batch of sampled dataset

'''

test_num = 10

def main():
    for i in range(test_num):
        #dis1 = new_alg(i)
        #print("New Algorithm: " + str(np.mean(dis1)) + "(" + str(np.var(dis1)) + ")")
        dis2 = process_MR(i, "MRW")
        print("MRW: " + str(np.mean(dis2)) + "(" + str(np.var(dis2)) + ")")


main()