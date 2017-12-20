import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw
data = np.load('detrend_gaussian10_dataset_X.npy')

x = data
y = data[3]
dist = []
for i in range(x.shape[0]):
    distance, path = fastdtw(x[i], y, dist=euclidean)
    dist.append(distance)
dist = np.array(dist)
print(dist)
