import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from config import random_seed


def auprc_score(estimator, X, y):
    y_pred = estimator.predict_proba(X)
    return average_precision_score(y, y_pred[:, 1])


steps = [
    ('rfe', RFECV(xgb.XGBClassifier(), step=100, cv=5, verbose=1000, scoring=auprc_score, )),
    ('xgb', xgb.XGBClassifier(n_estimators=1000, silent=True, nthread=3, seed=random_seed))
]

model = Pipeline(steps=steps)
