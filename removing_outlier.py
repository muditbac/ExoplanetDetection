import numpy as np
import matplotlib.pylab as plt
from PyAstronomy import pyasl
a = np.load('datasets/raw_normalized_dataset_X.npy')
from utils.processing_helper import save_features
median = np.median(a,axis=1)
std = np.std(a,axis=1)
x_new =[]
x = np.copy(a)
for i in xrange(x.shape[0]):
    index=[]
    r = pyasl.generalizedESD(x[i], 100, 2, fullOutput=True)
    r_1 = np.array(r[1])

    for j in xrange(x.shape[1]):
        if x[i][j] > median[i] + 2*std[i]:
            index.append(j)
    index = np.array(index)
    
    for k in np.intersect1d(index, r_1):
        x[i][k] = x[i][k-1]
    x_new.append(x[i])
x_new = np.array(x_new)
save_features(x_new,'raw_normalized_outliers_removed')
