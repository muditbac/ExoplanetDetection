from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import xgboost as xgb

steps = [
    ('pca', PCA(n_components=55)),
    ('xgb', xgb.XGBClassifier(n_estimators=1000, silent=True, n_jobs=1))
]
model = Pipeline(steps=steps)
