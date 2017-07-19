import numpy as np
import random
import pickle
from find_Neighbour_Users_beta import find_Neighbour_Users_beta
from find_Neighbour_Items_beta import find_Neighbour_Items_beta
from find_B_beta_LA import find_B_beta_LA


def preprocess_LA(num, beta):
    find_Neighbour_Users_beta(num, beta)
    find_Neighbour_Items_beta(num, beta)
    find_B_beta_LA(num)