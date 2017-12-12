from sklearn.svm import SVC
from config import random_seed
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

steps = [
    # ('pca', PCA(n_components = 55, random_state=random_seed)),
    ('SVC', SVC(C = 1, kernel = 'linear', random_state = random_seed, probability =True) )
]
model = Pipeline(steps=steps)
