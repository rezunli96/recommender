from cal_dis_New import cal_dis_New
from item_beat_New import item_beat_New
from find_Neighbour_New import find_Neighbour_New
from find_observed_rank_New import find_observed_rank_New
from cal_freq import cal_freq


def preprocess_new(num, beta, k , K):
    freq = cal_freq(num)
    find_observed_rank_New(num, freq)
    cal_dis_New(num, K)
    find_Neighbour_New(num, beta, k)
    item_beat_New(num)

