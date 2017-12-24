import argparse
import cPickle
import os

import numpy as np
import pandas as pd
from datetime import datetime

from config import TEST_PREDICTIONS_PATH, RESULTS_PATH
from utils.model_utils import load_model
from utils.processing_helper import load_testdata, make_dir_if_not_exists
from train_model import analyze_results
from utils.python_utils import start_logging

np.set_printoptions(precision=3)


def analyze_metrics(probs, target_filename):
    """
    Analyzes the predictions for other metrics
    :param probs: Predicted probabilities
    :param target_filename: Filename (csv) containing the labels
    """
    try:
        df = pd.read_csv(target_filename)
    except IOError:
        raise IOError("File %s doesnot exist !" % target_filename)
    target = df.values
    target = target[:, 0] - 1
    target = target.astype('int')
    print 'Analyzing other metrics for the predictions...'
    analyze_results(target, probs[:, 1])


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
    make_dir_if_not_exists(os.path.dirname(result_file))
    with open(result_file, 'wb') as fp:
        cPickle.dump(probs, fp)


def test_model(model_name, dataset_name, true_labels_path):
    """
    Loads and tests a pretrained model
    :param model_name: Name of the model to test
    :param dataset_name: Name of the dataset
    :param true_labels_path: CSV File path to containing true labels of test set
    """
    model = load_model(dataset_name, model_name)
    X = load_testdata(dataset_name)
    probs = model.predict_proba(X)
    print 'Saved the predicted probabilities'
    dump_results(probs, model_name, dataset_name)
    if true_labels_path:
        analyze_metrics(probs, true_labels_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset corresponding to given model')
    parser.add_argument('model', type=str, help="name of the py")
    parser.add_argument('--target', type=str, default=None, help="location to the file (csv) containing ground truth")
    args = parser.parse_args()

    # Log the output to file also
    current_timestring = datetime.now().strftime("%Y%m%d%H%M%S")
    start_logging(os.path.join(RESULTS_PATH, 'test_%s_%s_%s.txt' % (current_timestring, args.dataset, args.model)))

    test_model(args.model, args.dataset, args.target)
