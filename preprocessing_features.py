import os
import sys

import pandas as pd
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

from tsfresh.feature_extraction.feature_calculators import *

from config import *
from utils.processing_helper import save_features

def generate_time_series(x, y, name='time_series'):
    """Generates time series data
    """
    final_features = []
    functions = []

    functions.append(mean)
    functions.append(median)
    functions.append(length)
    functions.append(minimum)
    functions.append(maximum)
    functions.append(variance)
    functions.append(skewness)
    functions.append(kurtosis)
    functions.append(sum_values)
    functions.append(abs_energy)
    functions.append(mean_change)
    functions.append(has_duplicate)
    functions.append(sample_entropy)
    functions.append(mean_abs_change)
    functions.append(count_below_mean)
    functions.append(count_above_mean)
    functions.append(has_duplicate_min)
    functions.append(has_duplicate_max)
    functions.append(standard_deviation)
    functions.append(absolute_sum_of_changes)
    functions.append(last_location_of_minimum)
    functions.append(last_location_of_maximum)
    functions.append(first_location_of_maximum)
    functions.append(longest_strike_below_mean)
    functions.append(longest_strike_above_mean)
    functions.append(sum_of_reoccurring_values)
    functions.append(first_location_of_minimum)
    functions.append(sum_of_reoccurring_data_points)
    functions.append(variance_larger_than_standard_deviation)
    functions.append(ratio_value_number_to_time_series_length)
    functions.append(percentage_of_reoccurring_values_to_all_values)
    functions.append(percentage_of_reoccurring_datapoints_to_all_datapoints)

    for f in functions:
        sys.stdout.write('\r')
        sys.stdout.write('Generating feature: {}'.format(f.func_name))
        new_feature = []
        for k in xrange(x.shape[0]):
            z = f(x[k, :])
            new_feature.append(z)
        final_features.append(new_feature)
        sys.stdout.flush()
    sys.stdout.write('\nDone...\nDumping features...\n')
    final_features = np.array(final_features).T
    save_features(final_features, 'time_series')
    print 'Final features shape: '.format(final_features.shape)

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
    x, y = preprocess_data(dataset)
    if sys.argv[1] == 'time_series':
        print('Generating Time Series data...')
        generate_time_series(x, y)
        exit()
    x_smoothed = uniform_filter1d(x, axis=1, size=200)
    print("Dumping features...")
    save_features(x, 'raw_mean_std_normalized')
    save_features(x_smoothed, 'raw_mean_std_normalized_smoothed')
    y.dump(os.path.join(FEATURES_PATH, 'labels.npy'))

