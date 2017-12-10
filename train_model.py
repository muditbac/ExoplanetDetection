import argparse
import os
import pprint

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support

from config import RESULTS_PATH
from utils.processing_helper import load_dataset, load_folds

np.set_printoptions(precision=3)


def analyze_results(y_true, y_pred):
    """
    Prints the analysis of the predicted results
    :param y_true: True values
    :param y_pred: Predicted values
    """
    print "Overall AUC", roc_auc_score(y_true, y_pred), "\n"
    print "{:>15s} {:>15s} {:>15s} {:>15s} {:>15s}"\
        .format("Threshold", "Precision", "Recall", "F1-Score", "Skill Score")
    for threshold in np.linspace(0, 1, 21):
        cm = confusion_matrix(y_true, y_pred > threshold)
        skill_score = np.linalg.det(cm) / np.prod(np.sum(cm, axis=1))
        prec, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred > threshold)
        print "{:13.4f}   {:13.4f}   {:13.4f}   {:13.4f}   {:13.4f}"\
            .format(threshold, prec[1], recall[1], f1_score[1], skill_score)
    print


def summarize_model(model_name, dataset_name):
    """
    Trains the model with k-fold cross-validation and then generate the summary.
    Similar to sklearn.model_selection.cross_val_predict
    :param model_name: filename  in model package representing the pipeline or model object
    :param dataset_name: dataset name to load
    """
    model = __import__("models.%s" % model_name, globals(), locals(), ['model']).model

    X, y = load_dataset(dataset_name)
    y_complete_pred = np.zeros_like(y).astype('float')
    folds = load_folds()

    for i, (train_index, val_index) in enumerate(folds):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]

        # Copying the values to generate predictions of complete dataset
        y_complete_pred[val_index] = y_pred

        print "[Fold %d]: " % (i + 1),
        print "AUC: %5.4f" % roc_auc_score(y_val, y_pred)

    y_complete_pred.dump(os.path.join(RESULTS_PATH, '%s_%s.npy' % (dataset_name, model_name)))

    print
    analyze_results(y, y_complete_pred)
    print "\nModel parameters: "
    pprint.pprint(model.get_params(), indent=4, depth=1)
    print


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="The dataset to process")
    parser.add_argument('model', type=str, help="name of the py")
    args = parser.parse_args()

    summarize_model(args.model, args.dataset)
