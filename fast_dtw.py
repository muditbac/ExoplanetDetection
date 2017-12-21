import numpy as np
from scipy.spatial.distance import euclidean
from utils.processing_helper import save_features
from fastdtw import fastdtw
def dtw(data):
    x = data
    y = data[3]
    dist = []
    for i in range(x.shape[0]):
        distance, path = fastdtw(x[i], y, dist=euclidean)
        dist.append(distance)
    dist = np.array(dist)
    return dist
