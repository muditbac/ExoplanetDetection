import cPickle
import pickle
import numpy as np
import json
from config import FEATURES_PATH, DATASETS_PATH, FOLDS_FILENAME, MODELFILE_PATH
import os
from sklearn.base import BaseEstimator, TransformerMixin
import pickle as pkl
from keras.models import model_from_json

identity = lambda x: x


class SimpleTransform(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.transformer(X)


def generate_dataset(struct, dataset_name, is_test=False):
    """
    Generate the dataset from list of features and target variable
    :param struct: dict containing information of features and target variable
    :param dataset_name: name of the dataset
    :param is_test: if True then target values are not required in struct
    """
    features = struct['features']

    # Processing features
    features_numpy = []
    for i, (feature, transformer) in enumerate(features):
        x_all = np.load(os.path.join(FEATURES_PATH, '%s.npy' % feature))
        x_all = transformer.fit_transform(x_all)

        features_numpy.append(x_all)

    dataset = np.hstack(features_numpy)

    dataset_name = os.path.join(DATASETS_PATH, '%s_X.npy' % dataset_name)
    make_dir_if_not_exists(os.path.dirname(dataset_name))
    dataset.dump(dataset_name)

    if not is_test:
        target_feature_name, target_transformer = struct['target']
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


def load_testdata(dataset_name):
    """
    Loads test dataset
    """
    return np.load(os.path.join(DATASETS_PATH, 'test/%s_X.npy' % dataset_name))


def save_model(model, model_filename, cnn=False):
    """
    Saves the model
    :param model: model object
    :param model_filename: File name of the model
    """
    print 'Saving the model...'
    if not cnn:
        model_filename = os.path.join(MODELFILE_PATH, '%s.model' % model_filename)
        # make_dir_if_not_exists(os.path.dirname(model_filename))
        with open(model_filename, 'wb') as fp:
            pickle.dump(model, fp)
    else:
        json_model = model.model.to_json()
        model_filename_archi = os.path.join(MODELFILE_PATH, '%s_archi.model' % model_filename)
        model_filename_weights = os.path.join(MODELFILE_PATH, '%s_weights.model' % model_filename)
        # Save the architecture
        with open(model_filename_archi, 'wb') as f:
            f.write(json_model)
        # Save the weights
        model.model.save_weights(model_filename_weights, overwrite=True)


def load_model(dataset_name, model_file_name, cnn=False):
    """
    Loads a model
    :param model_file_name: Name of the model to load
    """
    if not cnn:
        with open(os.path.join(MODELFILE_PATH, dataset_name+'_'+model_file_name+'.model'), 'rb') as fp:
            return cPickle.load(fp)
    else:
        archi_file = os.path.join(MODELFILE_PATH, dataset_name+'_'+model_file_name+'_archi'+'.model')
        weights_file = os.path.join(MODELFILE_PATH, dataset_name+'_'+model_file_name+'_weights'+'.model')
        model = model_from_json(open(archi_file).read())
        model.load_weights(weights_file)
        return model


def save_features(data, features_name, test=False):
    """
    Save the features in the features folder
    :param data: numpy array or pandas dataframe with samples x features format
    :param features_name: name of the filename
    """
    if not isinstance(data, np.ndarray):
        data = data.values
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    if test:
        feature_filename = os.path.join(FEATURES_PATH, 'test/%s.npy' % features_name)
    else:
        feature_filename = os.path.join(FEATURES_PATH, '%s.npy' % features_name)
    make_dir_if_not_exists(os.path.dirname(feature_filename))
    data.dump(feature_filename)


def features_exists(feature_name, test=False):
    """
    Checks if feature/s exists with given name
    """
    if test:
        return os.path.exists(os.path.join(FEATURES_PATH, feature_name + '.npy'))
    else:
        return os.path.exists(os.path.join(FEATURES_PATH, 'test', feature_name + '.npy'))


def make_dir_if_not_exists(dir_name):
    """
    Makes directory if does not exists
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def load_folds():
    """
    Loads the k-folds split of the dataset for cross-validation
    :return: list of (train_split, test_split)
    """
    return pkl.load(open(FOLDS_FILENAME, 'r'))
