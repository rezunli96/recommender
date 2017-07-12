import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os



f = open("full_data.pkl", "rb")

a = pickle.load(f)

f.close()


print(a.shape)