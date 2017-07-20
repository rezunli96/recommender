import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os
import matplotlib.pyplot as plt


x = [0.2, 0.4, 0.6, 0.8]

y1 = [1.3662, 1.07032, 0.7, 0.3]
y2 = [1.4838, 1.17032, 0.8, 0.4]
plt.plot(x, y1, marker= "*", ls = "--", label = 'New')
plt.plot(x, y2, marker= "o", ls = "-.", label = 'MRW')
plt.legend()
plt.show()

