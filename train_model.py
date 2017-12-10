import argparse

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support

from utils.processing_helper import load_dataset, load_folds

np.set_printoptions(precision=3)


def analyze_results(y_true, y_pred, ):
    """
    TODO
    :param y_true:
    :param y_pred:
    """
    print "AUC", roc_auc_score(y_true, y_pred)
    for threshold in np.linspace(0, 1, 21):
        cm = confusion_matrix(y_true, y_pred > threshold)
        # print_cm(cm, ['not-exo', 'exo'])

        skill_score = np.linalg.det(cm) / np.prod(np.sum(cm, axis=1))
        prec, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred > threshold)
        print "Threshold, ", "%f %f, %f, %f, %f" % (threshold, prec[1], recall[1], f1_score[1], skill_score)
        print


def summarize_model(model, dataset_name):
    """
    Trains the model with k-fold cross-validation and then generate the summary
    :param model: the model or pipeline to fit and predict
    :param dataset_name: dataset name to load
    """
    model = __import__(model, globals(), locals(), ['model']).model

    X, y = load_dataset(dataset_name)
    y_complete_pred = np.zeros_like(y).astype('float')
    folds = load_folds()

    print "Model parameters: "
    print model.get_params()  # TODO Pretty print

    for i, (train_index, val_index) in enumerate(folds):
        print "Processing Fold %d" % (i + 1)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]

        # Copying the values to generate a complete dataset
        y_complete_pred[val_index] = y_pred

    analyze_results(y, y_complete_pred)
    # TODO Save the predicted values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="The dataset to process")
    parser.add_argument('model', type=str, help="name of the py")
    args = parser.parse_args()

    summarize_model(args.model, args.dataset)
