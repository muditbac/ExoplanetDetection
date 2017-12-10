import pandas as pd
import os

from config import *
from utils.processing_helper import save_features


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

    print("Dumping features...")

    save_features(x, 'raw_mean_std_normalized')
    y.dump(os.path.join(FEATURES_PATH, 'labels.npy'))
