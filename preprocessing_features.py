import numpy as np
import os
import pandas as pd
import scipy
import argparse

from config import *
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import medfilt, savgol_filter, find_peaks_cwt
from sklearn.preprocessing import MinMaxScaler

from utils.processing_helper import save_features
from pywt import dwt

PEAK_FEATURES_SIZE = 18

def get_spectrum(X):
    """
    Returns the spectrum of the given time series
    """
    spectrum = scipy.fft(X)
    return np.abs(spectrum)


def detrend_data(dataset_x, sigma=10):
    """
    Removes the trend from the time series using gaussian smoothing
    :param dataset_x:
    :param sigma:
    :return:
    """
    smooth_data = gaussian_filter(dataset_x, sigma=sigma)
    difference = dataset_x - smooth_data
    return difference


def detrend_data_median(dataset_x, kernel_size=81):
    smooth_data = dataset_x.apply(medfilt, axis=1, kernel_size=kernel_size)
    difference = dataset_x - smooth_data
    return difference.values


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


def peak_features(X):
    """
    Generates peak features
    """
    final_features = np.empty([len(X), PEAK_FEATURES_SIZE])

    def find_peaks(smooth):
        peaks = find_peaks_cwt(-smooth, np.arange(1, 100))
        return peaks

    def smoothen(data):
        return savgol_filter(data, polyorder=5, window_length=35)

    def dist_time(data):
        dist = data - np.roll(data, shift=-1)
        return np.mean(dist), np.std(dist)

    for idx, data in enumerate(X):
        smooth = smoothen(data)
        peaks = find_peaks(smooth)
        features = np.empty(PEAK_FEATURES_SIZE)

        std1, mean  = np.std(smooth), np.mean(smooth)
        std2, std3  = 2*std1, 3*std1
        min1  = np.where(smooth < (mean-std1))
        min2  = np.where(smooth < (mean-std2))
        min3  = np.where(smooth < (mean-std3))
        min01 = np.where(np.logical_and(smooth > (mean-std1), smooth < mean))
        min12 = np.where(np.logical_and(smooth < (mean-std1), smooth > (mean-std2)))
        min23 = np.where(np.logical_and(smooth < (mean-std2), smooth > (mean-std3)))
        
        # Compute the features
        features[0], features[1], features[2] = len(min1), len(min2), len(min3)
        features[3], features[4], features[5] = len(min01), len(min12), len(min23)
        features[6], features[7]   = dist_time(min1)
        features[8], features[9]   = dist_time(min2)
        features[10], features[11] = dist_time(min3)
        features[12], features[13] = dist_time(min1)
        features[14], features[15] = dist_time(min2)
        features[16], features[17] = dist_time(min3)

        final_features[idx] = features

    return final_features


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
        dataset_x = dataset
    else:
        dataset = pd.read_csv(raw_data_filename)
        dataset_x = dataset.iloc[:, 1:]

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
    for sigma in [5, 10, 15, 20]:
        x_detrend_sigma = detrend_data(dataset_x, sigma=sigma)
        save_features(x_detrend_sigma, 'detrend_gaussian%d' % sigma, args.test)

        fft_normalized_sigma = generate_fft_features(x_detrend_sigma)
        save_features(fft_normalized_sigma, 'fft_smoothed_sigma%d' % sigma, args.test)

    print(" - Detrending using median")
    for kernel_size in [21, 41, 81]:
        print(" - \t Processing for kernel size %d" % kernel_size)
        x_detrend_median = detrend_data_median(dataset_x, kernel_size=kernel_size)
        save_features(x_detrend_median, 'detrend_median%d' % kernel_size, args.test)

        fft_detrend_median = generate_fft_features(x_detrend_median)
        save_features(fft_detrend_median, 'fft_smoothed_median%d' % kernel_size, args.test)

    print ' - Processing Wavelet Features'
    wavelet_db2_a, wavelet_db2_b = dwt(x, 'db2')
    save_features(wavelet_db2_a, 'wavelet_db2_a', args.test)
    save_features(wavelet_db2_b, 'wavelet_db2_b', args.test)

    print ' - Peak Analysis features'
    peak_feats = peak_features(x)
    save_features(peak_feats, 'peak_features', args.test)
