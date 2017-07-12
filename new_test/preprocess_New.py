from cal_dis import cal_dis
from find_N import find_N
from train_item_beat import item_beat


def preprocess_new(num, beta):
    cal_dis(num)
    find_N(num, beta)
    item_beat(num)
