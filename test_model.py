import argparse
import cPickle
import os
import time

import numpy as np

from config import PROBS_PATH
from utils.processing_helper import load_dataset, load_testdata, make_dir_if_not_exists
from utils.model_saving import load_model

np.set_printoptions(precision=3)


def clear_prior_probs():
    """
    Removes any prior probabilities present
    """
    files = os.listdir(PROBS_PATH)
    for file in files:
        os.remove(os.path.join(PROBS_PATH, file))


def dump_results(probs, model_name, dataset_name):
    """
    Dumps the probabilities to a file
    :param probs: predicted probabilities
    :param model_name: Name of the model
    :param dataset_name: Name of the dataset
    """
    make_dir_if_not_exists(PROBS_PATH)
    result_file = os.path.join(PROBS_PATH, '{}_{}.probs'.format(model_name, dataset_name))
    with open(result_file, 'wb') as fp:
        cPickle.dump(probs, fp)


def test_model(model_name, dataset_name):
    """
    Loads and tests a pretrained model
    :param model_name: Name of the model to test
    :param dataset_name: Name of the dataset
    """
    clear_prior_probs()
    if model_name.split('_')[0] == 'cnn':
        model = load_model(dataset_name, model_name, cnn=True)
    else:
        model = load_model(dataset_name, model_name)
    X = load_testdata(dataset_name)
    probs = model.predict_proba(X)
    print 'Saved the predicted probabilities'
    dump_results(probs, model_name, dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset corresponding to given model')
    parser.add_argument('model', type=str, help="name of the py")
    args = parser.parse_args()

    test_model(args.model, args.dataset)