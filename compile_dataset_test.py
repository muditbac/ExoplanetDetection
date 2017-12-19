from utils.processing_helper import SimpleTransform, generate_dataset

if __name__ == '__main__':
    struct = {
        'features': [
            ('test/raw_mean_std_normalized_test', SimpleTransform()),
            ('test/raw_mean_std_normalized_smoothed_uniform200_test', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'raw_normalized_smoothed_dataset_test', True)

    # Raw data with gaussian smoothing 50
    struct = {
        'features': [
            ('test/raw_mean_std_normalized_test', SimpleTransform()),
            ('test/raw_mean_std_normalized_smoothed_gaussian50_test', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'raw_normalized_gaussian50_dataset_test', True)

    struct = {
        'features': [
            ('test/detrend_gaussian10_test', SimpleTransform()),
            ('test/raw_mean_std_normalized_smoothed_gaussian50_test', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'detrend_gaussian10_with_smoothed_dataset_test', True)

    struct = {
        'features': [
            ('test/detrend_gaussian10_test', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'detrend_gaussian10_dataset_test', True)

    struct = {
        'features': [
            ('test/detrend_gaussian5_test', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'detrend_gaussian5_dataset_test', True)

    struct = {
        'features': [
            ('test/detrend_gaussian15_test', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'detrend_gaussian15_dataset_test', True)

    struct = {
        'features': [
            ('test/fft_smoothed_sigma10_test', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'fft_smoothed10_dataset_test', True)

    struct = {
        'features': [
             ('test/fft_smoothed_sigma20_test', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'fft_smoothed20_dataset_test', True)
