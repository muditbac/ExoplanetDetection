import argparse
import os
from datetime import datetime

from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.model_selection import cross_val_predict

from config import RESULTS_PATH
from train_model import analyze_results
from utils.processing_helper import load_dataset, load_folds
from utils.python_utils import start_logging, print_dict


def tune_model(model_name, dataset_name, n_trials):
    module_file = __import__("models.%s" % model_name, globals(), locals(), ['model', 'params_space'])
    model = module_file.model
    params_space = module_file.params_space

    X, y = load_dataset(dataset_name)
    folds = load_folds()

    def objective_fn(params):
        print "\n\nHyper-Parameters: "
        print_dict(params)

        model.set_params(**params)
        y_pred = cross_val_predict(model, X, y, cv=folds, method='predict_proba', verbose=2, n_jobs=2)
        results = analyze_results(y, y_pred[:, 1])
        max_f1_score = max(results, key=lambda x: x[1][3])[1][3]
        return {'loss': -max_f1_score, 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(fn=objective_fn,
                       space=params_space,
                       algo=tpe.suggest,
                       max_evals=n_trials,
                       trials=trials
                       )
    best_params['z_score'] = -trials.best_trial['result']['loss']
    print "\n\nBest Parameters..."
    print_dict(best_params)
    return best_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="The dataset to process")
    parser.add_argument('model', type=str, help="name of the python file")
    parser.add_argument('--trials', '-t', type=int, default=3, help="Number of trials to choose the perform the "
                                                                    "validation")
    args = parser.parse_args()

    # Log the output to file also
    current_timestring = datetime.now().strftime("%Y%m%d%H%M%S")
    start_logging(os.path.join(RESULTS_PATH, 'tune_%s_%s_%s.txt' % (current_timestring, args.dataset, args.model)))

    tune_model(args.model, args.dataset, n_trials=args.trials)
