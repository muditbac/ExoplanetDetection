from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from config import random_seed

steps = [
    ('lle', LocallyLinearEmbedding(n_components=55, n_jobs=4,  random_state=random_seed)),
    ['knn', KNeighborsClassifier(n_neighbors=10, n_jobs=4)]
]
model = Pipeline(steps=steps)
