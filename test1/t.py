
import numpy as np
import pickle
from scipy.sparse import lil_matrix
# jit decorator tells Numba to compile this function.

n2 = 10
ken = np.zeros(n2)
s = np.zeros(n2)
nd = np.zeros(n2)
p = np.zeros(n2)

ken_pkl = open("ken_pkl.pkl", "wb")
s_pkl = open("s_pkl.pkl", "wb")
nd_pkl = open("nd_pkl.pkl", "wb")
p_pkl = open("p_pkl.pkl", "wb")

pickle.dump(ken, ken_pkl)
pickle.dump(s, s_pkl)
pickle.dump(nd, nd_pkl)
pickle.dump(p, p_pkl)

ken_pkl.close()
s_pkl.close()
nd_pkl.close()
p_pkl.close()
f = open("res_num_MR.txt", 'w')

f.write("kendall_tau: "+str(float(np.mean(ken)))+"("+str(float(np.var(ken)))+")\n")
f.write("pearman_rho: "+str(float(np.mean(s)))+"("+str(float(np.var(s)))+")\n")
f.write("NDCG: "+str(float(np.mean(nd)))+"("+str(float(np.var(nd)))+")\n")
f.write("Precision: "+str(float(np.mean(p)))+"("+str(float(np.var(p)))+")\n")

f.close()
#print(s_iu)

#print(s_ui)