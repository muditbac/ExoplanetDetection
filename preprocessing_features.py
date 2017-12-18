import numpy as np
import os
import pandas as pd
import scipy
import argparse

from config import *
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import uniform_filter1d
from sklearn.preprocessing import MinMaxScaler

from utils.processing_helper import save_features


def get_spectrum(X):
    """
    Returns the spectrum of the given time series
    """
    spectrum = scipy.fft(X)
    return np.abs(spectrum)


def detrend_data(raw_dataset, sigma=10):
    """
    Removes the trend from the time series using gaussian smoothing
    :param raw_dataset:
    :param sigma:
    :return:
    """
    data = raw_dataset.values[:, 1:]
    smooth_data = gaussian_filter(data, sigma=sigma)
    difference = data - smooth_data
    return difference


def generate_fft_features(raw_signals):
    """
    Generates the FFT features of the detrended series from the dataset
    """
    scaler = MinMaxScaler()
    difference = np.transpose(raw_signals)
    difference_normalized = scaler.fit_transform(difference)
    difference_normalized = np.transpose(difference_normalized)
    fft_abs = get_spectrum(difference_normalized)
    half_length = (difference.shape[0] + 1) // 2
    return fft_abs[:, :half_length]


def preprocess_data(raw_data, is_test=False):
    """
    Simple pre-processing of the initial raw data and test data
    :param raw_data: initial raw data from CSV file
    :param is_test: bool for processing test data
    :return: features, labels
        features: the flux readings of the KOIs
        labels: 0 as non-exoplanet and 1 as exoplanet
    """
    if isinstance(raw_data, pd.DataFrame):
        raw_data = raw_data.values
    if not is_test:
        labels = raw_data[:, 0] - 1
        features = raw_data[:, 1:]
    else:
        features = raw_data
    mean = features.mean(axis=1).reshape(-1, 1)
    std = features.std(axis=1).reshape(-1, 1)
    features = (features - mean) / std

    if not is_test:
        return features, labels.astype('int')
    else:
        return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, help="'train' or 'test' data", default=False)
    args = parser.parse_args()
    if args.test:
        dataset = pd.read_csv(testing_filename)
    else:
        dataset = pd.read_csv(raw_data_filename)

    print("Preprocessing data...")

    print(" - Normalizing data")
    if not args.test
        x, y = preprocess_data(dataset)
    else:
        x = preprocess_data(dataset, True)

    print(" - Smoothing features")
    x_smoothed_uniform = uniform_filter1d(x, axis=1, size=200)
    x_smoothed_gaussian = gaussian_filter(x, sigma=50)

    print(" - Saving calculated features")
    feature_name = 'raw_mean_std_normalized'
    feature_name = feature_name + '_test' if args.test else feature_name 
    save_features(x, feature_name)

    feature_name = 'raw_mean_std_normalized_smoothed_uniform200'
    feature_name = feature_name + '_test' if  args.test else feature_name 
    save_features(x_smoothed_uniform, feature_name)
    
    feature_name = 'raw_mean_std_normalized_smoothed_gaussian50'
    feature_name = feature_name + '_test' if args.test else feature_name 
    save_features(x_smoothed_gaussian, feature_name)

    print(" - Detrending data")
    x_detrend_sigma15 = detrend_data(dataset, sigma=15)
    x_detrend_sigma10 = detrend_data(dataset, sigma=10)
    x_detrend_sigma5 = detrend_data(dataset, sigma=5)

    feature_name = 'detrend_gaussian15'
    feature_name = feature_name + '_test' if args.test else feature_name 
    save_features(x_detrend_sigma15, feature_name)

    feature_name = 'detrend_gaussian10'
    feature_name = feature_name + '_test' if args.test else feature_name 
    save_features(x_detrend_sigma10, feature_name)

    feature_name = 'detrend_gaussian5'
    feature_name = feature_name + '_test' if args.test else feature_name 
    save_features(x_detrend_sigma5, feature_name)

    print " - Generating and Saving FFT Features"
    fft_normalized_15 = generate_fft_features(x_detrend_sigma15)
    fft_normalized_10 = generate_fft_features(x_detrend_sigma10)
    fft_normalized_5 = generate_fft_features(x_detrend_sigma5)

    feature_name = 'fft_smoothed_sigma15'
    feature_name = feature_name + '_test' if args.test else feature_name 
    save_features(fft_normalized_15, feature_name)

    feature_name = 'fft_smoothed_sigma10'
    feature_name = feature_name + '_test' if args.test else feature_name 
    save_features(fft_normalized_10, feature_name)

    feature_name = 'fft_smoothed_sigma5'
    feature_name = feature_name + '_test' if args.test else feature_name 
    save_features(fft_normalized_5, feature_name)

    if not args.test:
        y.dump(os.path.join(FEATURES_PATH, 'labels.npy'))
