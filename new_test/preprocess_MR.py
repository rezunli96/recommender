import numpy as np
import random
import pickle
from N_uv import N
from Nuv_candidate import N_C
from R_uv import R


def preprocess_MR(num):
    N(num)
    N_C(num)
    R(num)
