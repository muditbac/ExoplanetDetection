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
    struct = {
        'features': [
            # ('probability_results/wavelet_db2_b_dataset_xgb', ReshapeTransform()),
            ('probability_results/raw_normalized_gaussian50_dataset_cnn_wrapper_2d_tune', ReshapeTransform()),
            # ('probability_results/raw_normalized_smoothed_dataset_cnn_wrapper_2d_tune', ReshapeTransform()),
            ('probability_results/raw_normalized_smoothed_dataset_cnn_wrapper_window_slicing_size1000', ReshapeTransform()),
            # ('probability_results/detrend_gaussian10_dataset_edited_nn_pca_xgb_tune', ReshapeTransform()),
            ('probability_results/fft_smoothed10_dataset_xgb_tune', ReshapeTransform()),
            ('probability_results/fft_smoothed10_dataset_edited_nn_pca_xgb_tune', ReshapeTransform()),
            ('probability_results/fft_smoothed10_dataset_cnn_wrapper_1d_half', ReshapeTransform()),
            # ('probability_results/fft_normalized_dataset_onesided_pca_xgb_tune', ReshapeTransform()),
            # ('probability_results/fft_smoothed10_dataset_lle_xgb_tune', ReshapeTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'ensemble_dataset_dummy')
