import numpy as np
import xgboost as xgb
from hyperopt import hp
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from config import random_seed
from utils.python_utils import quniform_int

steps = [
    ('pca', PCA(n_components=55, random_state=random_seed)),
    ('xgb', xgb.XGBClassifier(n_estimators=1000, silent=True, nthread=1, seed=random_seed))
]
model = Pipeline(steps=steps)

params_space = {
    'xgb__max_depth': hp.choice('max_depth', np.arange(10, 30, dtype=int)),
    'xgb__min_child_weight': quniform_int('min_child', 1, 20, 1),
    'xgb__subsample': hp.uniform('subsample', 0.8, 1),
    'xgb__n_estimators': hp.choice('n_estimators', np.arange(1000, 10000, 100, dtype=int)),
    'xgb__learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
    'xgb__gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'xgb__colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05)
}
