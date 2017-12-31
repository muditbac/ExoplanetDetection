import argparse

import numpy as np
from utils.processing_helper import generate_dataset, SimpleTransform
from sklearn.base import BaseEstimator, TransformerMixin


class ReshapeTransform(BaseEstimator, TransformerMixin):
    def __init__(self, reshape_size=(-1, 1)):
        self.reshape_size = reshape_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(self.reshape_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="'train' or 'test' data", action='store_true')
    args = parser.parse_args()

    struct = {
        'features': [
            ('probs/raw_normalized_gaussian50_dataset_cnn_wrapper_2d_rng50', ReshapeTransform()),
            ('probs/raw_normalized_smoothed_dataset_cnn_window_slicing_2d_rns', ReshapeTransform()),
            # ('probs/fft_smoothed10_dataset_xgb_fft10', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_edited_nn_pca_xgb_fft10', ReshapeTransform()),
            ('probs/fft_smoothed10_dataset_cnn_wrapper_fft', ReshapeTransform()),
            # ('probs/fft_smoothed10_dataset_cnn_wrapper_1d_half_no_rolling_fft10', ReshapeTransform()),
            ('probs/raw_normalized_gaussian50_dataset_xgb', ReshapeTransform()),
            ('probs/fft_smoothed10_ar100_dataset_edited_nn_xgb_fft10ar100', ReshapeTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'ensembling_dataset_dummy', test=args.test)
