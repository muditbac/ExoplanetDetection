import argparse
import cPickle
import os

import numpy as np
import pandas as pd

from config import TEST_PREDICTIONS_PATH
from utils.model_utils import load_model
from utils.processing_helper import load_testdata, make_dir_if_not_exists
from train_model import analyze_results

np.set_printoptions(precision=3)

def analyze_metrics(probs, target_file):
    """
    Analyzes the predictions for other metrics
    :param probs: Predicted probabilities
    :param target_file: File (csv) containing the labels
    """
    try:
        df = pd.read_csv(target_file)
    except IOError:
        raise IOError("File %s doesnot exist !" % target_file)
    target = df.values
    target = target[:, 1] - 1
    target = target.astype(np.int)
    print 'Analyzing other metrics for the predictions...'
    analyze_results(target, probs[:,1])


def clear_prior_probs():
    """
    Removes any prior probabilities present
    """
    make_dir_if_not_exists(TEST_PREDICTIONS_PATH)
    files = os.listdir(TEST_PREDICTIONS_PATH)
    for file_name in files:
        os.remove(os.path.join(TEST_PREDICTIONS_PATH, file_name))


def dump_results(probs, model_name, dataset_name):
    """
    Dumps the probabilities to a file
    :param probs: predicted probabilities
    :param model_name: Name of the model
    :param dataset_name: Name of the dataset
    """
    result_file = os.path.join(TEST_PREDICTIONS_PATH, '{}_{}.probs'.format(model_name, dataset_name))
    with open(result_file, 'wb') as fp:
        cPickle.dump(probs, fp)


def test_model(model_name, dataset_name, target_file):
    """
    Loads and tests a pretrained model
    :param model_name: Name of the model to test
    :param dataset_name: Name of the dataset
    """
    model = load_model(dataset_name, model_name)
    X = load_testdata(dataset_name)
    probs = model.predict_proba(X)
    print 'Saved the predicted probabilities'
    dump_results(probs, model_name, dataset_name)
    if not target_file :
        analyze_metrics(probs, target_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset corresponding to given model')
    parser.add_argument('model', type=str, help="name of the py")
    parser.add_argument('--target', type=str, default='', help="location to the file (csv) containing ground truth")
    args = parser.parse_args()

    test_model(args.model, args.dataset, args.target)
