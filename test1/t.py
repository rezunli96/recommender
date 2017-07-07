import numpy as np
import pickle
from scipy.sparse import lil_matrix
import random
# jit decorator tells Numba to compile this function.
import os

n1 = 3952
n2 = 6040

def change(a):

    a = list(reversed(a))
    print(a)


a = [0, 1, 2, 3, 4, 5]

print(a)

change(a)

print(a)



