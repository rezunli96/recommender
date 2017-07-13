import numpy as np
import random
import pickle
from N_uv import N
from Nuv_candidate import N_C
from R_uv import R_C


def preprocess_MR(num, beta):
    N_C(num, beta)
    R_C(num)
