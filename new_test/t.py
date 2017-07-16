import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os
import matplotlib.pyplot as plt

f = open("new_to_K.pkl", "rb")

opt_res_new = pickle.load(f)

f.close()

f = open("mr_to_K.pkl", "rb")

opt_res_mr = pickle.load(f)

f.close()

plt.figure()
plt.plot([x[0] for x in opt_res_new], [x[1] for x in opt_res_new])
plt.plot([x[0] for x in opt_res_mr], [x[1] for x in opt_res_mr])
print(opt_res_new)
print(opt_res_mr)
plt.show()
