import argparse
import os
from datetime import datetime

import numpy as np
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier

from config import RESULTS_PATH
from config import random_seed
from utils.processing_helper import load_dataset
from utils.python_utils import start_logging

np.set_printoptions(precision=4)


def xgboost_importance(dataset_name):
    """
    Trains a XGBClassifier on the dataset and generates the feature importance.
    :param dataset_name: dataset name to load
    """
    X, y = load_dataset(dataset_name)
    classifier = xgb.XGBClassifier(n_estimators=5900, max_depth=11, min_child_weight=4, subsample=0.932626370862,
                                   gamma=0.7, colsample_bytree=0.85, learning_rate=0.125, silent=True, nthread=1,
                                   seed=random_seed)
    classifier.fit(X, y)
    importances = classifier.feature_importances_
    return importances


def randomforest_importance(dataset_name):
    """
    Trains a RandomForestClassifier on the dataset and generates the feature importance.
    :param dataset_name: dataset name to load
    """
    X, y = load_dataset(dataset_name)

    classifier = RandomForestClassifier(random_state=random_seed)
    classifier.fit(X, y)
    importances = classifier.feature_importances_
    return importances


def univariate_importance(dataset_name):
    """
    Trains an LDA model on the dataset and generates the LDA coefficients for feature importance
    :param dataset_name: dataset name to load
    """
    X, y = load_dataset(dataset_name)

    classifier = LDA(store_covariance=True)
    classifier.fit(X, y)
    importances = classifier.coef_
    return importances


def generate_importance(dataset_name):
    """
    Generates the feature importance using different models
    :param dataset_name: dataset name to load
    """
    rf_imp = randomforest_importance(dataset_name)
    xgb_imp = xgboost_importance(dataset_name)
    lda_imp = univariate_importance(dataset_name)
    importance_table = np.vstack([rf_imp, xgb_imp, lda_imp])

    print importance_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="The dataset to process")
    args = parser.parse_args()

    # Log the output to file also
    current_timestring = datetime.now().strftime("%Y%m%d%H%M%S")
    start_logging(os.path.join(RESULTS_PATH, '%s_%s_%s.txt' % (current_timestring, args.dataset, 'feature_importance')))

    generate_importance(args.dataset)
