import numpy as np
import random
import pickle
from find_Neighbour_Users_beta import find_Neighbour_Users_beta
from find_R_MR import find_R_MR


def preprocess_MR(num, beta):
    find_Neighbour_Users_beta(num, beta)
    find_R_MR(num)
