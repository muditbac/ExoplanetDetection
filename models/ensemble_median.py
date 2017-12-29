import numpy as np
from sklearn.base import BaseEstimator


class MedianModel(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        y = (np.median(X, axis=1)+10*np.max(X, axis=1))/11
        return np.vstack([1 - y, y]).T

model = MedianModel()
