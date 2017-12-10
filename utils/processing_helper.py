import numpy as np
from config import FEATURES_PATH, DATASETS_PATH, FOLDS_FILENAME
import os
from sklearn.base import BaseEstimator, TransformerMixin
import pickle as pkl

identity = lambda x: x


class SimpleTransform(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.transformer(X)


def generate_dataset(struct, dataset_name):
    """
    Generate the dataset from list of features and target variable
    :param struct: dict containing information of features and target variable
    :param dataset_name: name of the dataset
    """
    features = struct['features']
    target_feature_name, target_transformer = struct['target']

    # Processing features
    features_numpy = []
    for i, (feature, transformer) in enumerate(features):
        x_all = np.load(os.path.join(FEATURES_PATH, '%s.npy' % feature))
        x_all = transformer.fit_transform(x_all)

        features_numpy.append(x_all)

    dataset = np.hstack(features_numpy)
    dataset.dump(os.path.join(DATASETS_PATH, '%s_X.npy' % dataset_name))

    # Processing the target variable
    target_values = np.load(os.path.join(FEATURES_PATH, '%s.npy' % target_feature_name))
    target_values = target_transformer.fit_transform(target_values)
    target_values.dump(os.path.join(DATASETS_PATH, '%s_y.npy' % dataset_name))


def load_dataset(dataset_name):
    """
    Loads the dataset from the dataset folds given the dataset name
    :param dataset_name
    :return: dataset_X, dataset_y
    """
    X = np.load(os.path.join(DATASETS_PATH, '%s_X.npy' % dataset_name))
    y = np.load(os.path.join(DATASETS_PATH, '%s_y.npy' % dataset_name))
    return X, y


def save_features(data, features_name):
    """
    Save the features in the features folder
    :param data: numpy array or pandas dataframe with samples x features format
    :param features_name: name of the filename
    """
    if not isinstance(data, np.ndarray):
        data = data.values
    if len(data.shape) == 1:
        data.reshape(-1, 1)
    data.dump(FEATURES_PATH + features_name + '.npy')


def load_folds():
    """
    Loads the k-folds split of the dataset for cross-validation
    :return: list of (train_split, test_split)
    """
    return pkl.load(open(FOLDS_FILENAME, 'r'))