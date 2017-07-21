import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os
import matplotlib.pyplot as plt


x = [0.2, 0.4, 0.6, 0.8]

y1 = [1.3975, 1.049, 0.7466, 0.3876]
y2 = [1.3462, 1.0308, 0.7394, 0.4]
y3 = [1.6252, 1.5724, 1.5324]
plt.plot(x, y1,  ls = "--", label = 'New')
plt.plot(x, y2,  ls = "-.", label = 'LA')
plt.legend()
plt.show()

