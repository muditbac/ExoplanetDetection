import argparse
import os
import time
import pprint
import cPickle

import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, average_precision_score

from config import RESULTS_PATH, PROBS_PATH
from utils.processing_helper import load_dataset, load_testdata
from train_model import load_model

np.set_printoptions(precision=3)

# Stores the {idx, [model, dataset]} pairs
model_data_dict = {
   0: ["cnn_wrapper_2d", "raw_normalized_gaussian50_dataset"],
   1: ["cnn_wrapper_window_slicing", "raw_normalized_smoothed_dataset"],
   2: ["xgb", "fft_smoothed10_dataset"],
   3: ["edited_nn_pca_xgb", "fft_smoothed10_dataset"],
   4: ["cnn_wrapper_1d", "fft_smoothed10_dataset"]
}

def dump_results(probs, model_name, dataset_name):
   """
   Dumps the probabilities to a file
   :param probs: predicted probabilities
   :param model_name: Name of the model
   :param dataset_name: Name of the dataset 
   """
   if not os.path.exists(PROBS_PATH):
      os.makedirs(PROBS_PATH)
   result_file = os.path.join(PROBS_PATH, '{}_{}_{}.probs'.format(int(time.time()), model_name, dataset_name))
   with open(result_file, 'wb') as fp:
      cPickle.dump(probs, fp)


def test_model(model_name, dataset_name):
   """
   Loads and tests a pretrained model
   :param model_name: Name of the model to test
   :param dataset_name: Name of the dataset
   """
   if model_name == "":
      final_probs = []
      for key, [model_name, dataset_name] in model_data_dict.iteritems():
         X = load_testdata(dataset_name)
         model = load_model(model_name)
         probs = model.predict_proba(X)[:, 1]
         final_probs.append(probs)
      final_probs = np.mean(np.array(final_probs), axis=0)
      dump_results(final_probs, 'ensemble_model', 'ensemble_dataset')
   else:
      X, _ = load_dataset(dataset_name)
      # X, _ = load_testdata(dataset_name)
      model = load_model(model_name)
      probs = model.predict_proba(X)[:, 1]
      print 'Saved the predicted probabilities'
      dump_results(probs, model_name, dataset_name)


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--dataset', type=str, help='dataset corresponding to given model {ensemble if not mentioned}', default="")
   parser.add_argument('--model', type=str, help="name of the py {ensemble if not mentioned}", default="")
   parser.add_argument('--ensemble', help="Should ensemble or not", action='store_true')
   args = parser.parse_args()

   test_model(args.model, args.dataset)
