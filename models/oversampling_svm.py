from sklearn.svm import SVC
from config import random_seed
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

steps = [
	('oversampler', RandomOverSampler(random_state = random_seed)),
    ('SVC', SVC(C = 1, kernel = 'linear', random_state = random_seed, probability =True) )
]
model = Pipeline(steps=steps)
