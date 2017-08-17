from utils import *
from LA import process_LA
from MR import process_MR
from RMC import process_RMC
from NEW import process_New_average_agg

'''

Main file to run different algorithm on batch of sampled dataset


'''


def search_for_NEW():
    for prob in p:
        opt_beta = 0
        opt_lam = 0
        opt = 99
        upper = min(4 * int(prob * prob * 100) + 3, 100)
        for beta in range(2, upper):
            for lam in range(0, 51, 2):
                l = lam * 0.5
                d = np.zeros(test_num)
                for num in range(test_num):
                    di = str(prob) + "\\" + str(num)
                    preprocess_new(di, beta=beta, k=100, K=K)
                    dis = process_New_average_agg(di, K, beta=beta, k=100, lam=l)
                    d[num] = np.mean(dis)
                if (np.mean(d) < opt):
                    opt = np.mean(d)
                    opt_beta = beta
                    opt_lam = l
                print((beta, l), np.mean(d))
        print("For prob",prob, "optimal for NEW is",(opt_beta,opt_lam),opt)
        '''
        f  = open(".\\result\\"+str(prob)+"\\opt_res_new.pkl", "wb")
        pickle.dump(res, f)
        f.close()
        '''


def search_for_LA():
    for prob in p:
        opt_beta = 0
        opt_lam = 0
        opt = 99
        upper = min(4 * int(prob * prob * 100)+3, 100)
        for beta in range(2, upper):
            for lam in range(0, 51, 2):
                l = lam * 0.1
                d = np.zeros(test_num)
                for num in range(test_num):
                    di = str(prob) + "\\" + str(num)
                    preprocess_LA(di, beta=beta)
                    dis  = process_LA(di, K, lam=l)
                    d[num] = np.mean(dis)
                    #print("In LA for ",(beta, l), "result is", np.mean(d))
                    #res.append((beta, l, np.mean(d)))
                if(np.mean(d) < opt):
                    opt = np.mean(d)
                    opt_beta = beta
                    opt_lam = l
                print((beta, l), np.mean(d))
        print("For prob",prob, "optimal for LA is",(opt_beta,opt_lam),opt)
        '''
        f  = open(".\\result\\"+str(prob)+"\\opt_res_LA.pkl", "wb")
        pickle.dump(res, f)
        f.close()
        '''

def search_for_RMC(max_iter):
    for prob in p:
        opt_epsilon = 0
        opt_tau = 0
        opt = 99
        for e in range(100):
            epsilon = e * 0.0001
            for t in range(0, 31):
                tau = t * 0.1
                d = np.zeros(test_num)
                for num in range(test_num):
                    di = str(prob) + "\\" + str(num)
                    dis  = process_RMC(di, epsilon, tau, max_iter, K)
                    d[num] = np.mean(dis)
                if(np.mean(d) < opt):
                    opt = np.mean(d)
                    opt_epsilon = epsilon
                    opt_tau = tau
                print((epsilon, tau), np.mean(d))
        print("For prob",prob, "optimal for RMC is",(opt_epsilon,opt_tau),opt)
        '''
        f  = open(".\\result\\"+str(prob)+"\\opt_res_RMC.pkl", "wb")
        pickle.dump(res, f)
        f.close()
        '''

def search_for_MRW(version):
    for prob in p:
        opt_beta = 0
        opt_k = 0
        opt = 99
        res = []
        for beta in range(100, 101):
            for k in range(5, 51, 5):
                d = np.zeros(test_num)
                for num in range(test_num):
                    di = str(prob) + "\\" + str(num)
                    preprocess_MR(di, beta=beta)
                    dis = process_MR(di, version, k=k, K=K)
                    d[num] = np.mean(dis)
                if (np.mean(d) < opt):
                    opt = np.mean(d)
                    opt_beta = beta
                    opt_k = k
                print((beta, k), np.mean(d))
        print("For prob", prob, "optimal for MRW is", (opt_beta, opt_k), opt)

        print("Optimal for MRW: ", (opt_beta, opt_k), opt)
        '''
        f  = open(".\\result\\"+str(num)+"\\opt_res_MRW.pkl", "wb")
        pickle.dump(res, f)
        f.close()
        '''



test_num = 5

K = 15

p = [0.1, 0.2, 0.3, 0.4]


def main():
    generate_all(test_num, p, "jester", 100)
    #search_for_LA()
    #search_for_MRW("MRW")
    #search_for_RMC(100)
    search_for_NEW()

main()