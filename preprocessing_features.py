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
from pywt import dwt

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
    parser.add_argument('--test', help="'train' or 'test' data", action='store_true')
    args = parser.parse_args()
    if args.test:
        dataset = pd.read_csv(testing_filename)
    else:
        dataset = pd.read_csv(raw_data_filename)

    print("Preprocessing data...")

    print(" - Normalizing data")
    if not args.test:
        x, y = preprocess_data(dataset)
        y.dump(os.path.join(FEATURES_PATH, 'labels.npy'))
    else:
        x = preprocess_data(dataset, True)

    print(" - Smoothing features")
    x_smoothed_uniform = uniform_filter1d(x, axis=1, size=200)
    x_smoothed_gaussian = gaussian_filter(x, sigma=50)

    print(" - Saving calculated features")
    save_features(x, 'raw_mean_std_normalized', args.test)
    save_features(x_smoothed_uniform, 'raw_mean_std_normalized_smoothed_uniform200', args.test)
    save_features(x_smoothed_gaussian, 'raw_mean_std_normalized_smoothed_gaussian50', args.test)

    print(" - Detrending and generating FFT features data")
    for sigma in [5, 10, 50]:
        x_detrend_sigma = detrend_data(dataset, sigma=sigma)
        save_features(x_detrend_sigma, 'detrend_gaussian%d' % sigma, args.test)

        fft_normalized_sigma = generate_fft_features(x_detrend_sigma)
        save_features(fft_normalized_sigma, 'fft_smoothed_sigma%d' % sigma, args.test)

    print ' - Processing Wavelet Features'
    wavelet_db2_a, wavelet_db2_b = dwt(x, 'db2')
    save_features(wavelet_db2_a, 'wavelet_db2_a')
    save_features(wavelet_db2_b, 'wavelet_db2_b')
