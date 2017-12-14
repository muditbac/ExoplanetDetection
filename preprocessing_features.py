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


def generate_fft_features(raw_dataset, sigma=10):
    """
    Generates the FFT features of the detrended series from the dataset
    """
    data = raw_dataset.values[:, 1:]
    smooth_data = gaussian_filter(data, sigma=sigma)
    difference = data - smooth_data
    scaler = MinMaxScaler()
    difference = np.transpose(difference)
    difference_normalized = scaler.fit_transform(difference)
    difference_normalized = np.transpose(difference_normalized)
    fft_abs = get_spectrum(difference_normalized)
    half_length = (data.shape[1] + 1) // 2
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
    x_smoothed_gaussian = gaussian_filter(x, sigma=20)

    print(" - Saving calculated features")
    save_features(x, 'raw_mean_std_normalized')
    save_features(x_smoothed_uniform, 'raw_mean_std_normalized_smoothed_uniform200')
    save_features(x_smoothed_gaussian, 'raw_mean_std_normalized_smoothed_gaussian20')

    print " - Generating and Saving FFT Features"
    fft_normalized = generate_fft_features(dataset, sigma=10)
    save_features(fft_normalized, 'fft_smoothed_sigma10')
    fft_normalized = generate_fft_features(dataset, sigma=20)
    save_features(fft_normalized, 'fft_smoothed_sigma20')

    y.dump(os.path.join(FEATURES_PATH, 'labels.npy'))
