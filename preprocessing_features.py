import numpy as np
import os
import pandas as pd
import scipy

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
    half_length = (difference.shape[1] + 1) // 2
    return fft_abs[:, :half_length]


def preprocess_data(raw_data):
    """
    Simple pre-processing of the initial raw data
    :param raw_data: initial raw data from CSV file
    :return: features, labels
        features: the flux readings of the KOIs
        labels: 0 as non-exoplanet and 1 as exoplanet
    """
    if isinstance(raw_data, pd.DataFrame):
        raw_data = raw_data.values
    labels = raw_data[:, 0] - 1
    features = raw_data[:, 1:]

    mean = features.mean(axis=1).reshape(-1, 1)
    std = features.std(axis=1).reshape(-1, 1)

    features = (features - mean) / std

    return features, labels.astype('int')


if __name__ == '__main__':
    dataset = pd.read_csv(raw_data_filename)

    print("Preprocessing data...")

    print(" - Normalizing data")
    x, y = preprocess_data(dataset)

    print(" - Smoothing features")
    x_smoothed_uniform = uniform_filter1d(x, axis=1, size=200)
    x_smoothed_gaussian = gaussian_filter(x, sigma=50)

    print(" - Saving calculated features")
    save_features(x, 'raw_mean_std_normalized')
    save_features(x_smoothed_uniform, 'raw_mean_std_normalized_smoothed_uniform200')
    save_features(x_smoothed_gaussian, 'raw_mean_std_normalized_smoothed_gaussian50')

    print(" - Detrending data")
    x_detrend_sigma15 = detrend_data(dataset, sigma=15)
    x_detrend_sigma10 = detrend_data(dataset, sigma=10)
    x_detrend_sigma5 = detrend_data(dataset, sigma=5)
    save_features(x_detrend_sigma15, 'detrend_gaussian15')
    save_features(x_detrend_sigma10, 'detrend_gaussian10')
    save_features(x_detrend_sigma5, 'detrend_gaussian5')

    print " - Generating and Saving FFT Features"
    fft_normalized_15 = generate_fft_features(x_detrend_sigma15)
    fft_normalized_10 = generate_fft_features(x_detrend_sigma10)
    fft_normalized_5 = generate_fft_features(x_detrend_sigma5)

    save_features(fft_normalized_15, 'fft_smoothed_sigma15')
    save_features(fft_normalized_10, 'fft_smoothed_sigma10')
    save_features(fft_normalized_5, 'fft_smoothed_sigma5')

    y.dump(os.path.join(FEATURES_PATH, 'labels.npy'))
