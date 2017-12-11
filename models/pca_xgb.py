from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import xgboost as xgb

from config import random_seed

steps = [
    ('pca', PCA(n_components=55, random_state=random_seed)),
    ('xgb', xgb.XGBClassifier(n_estimators=1000, silent=True, nthread=1, seed=random_seed))
]
model = Pipeline(steps=steps)
