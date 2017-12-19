from utils.processing_helper import SimpleTransform, generate_dataset

if __name__ == '__main__':
    struct = {
        'features': [
            ('test/raw_mean_std_normalized', SimpleTransform()),
            ('test/raw_mean_std_normalized_smoothed_uniform200', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'test/raw_normalized_smoothed_dataset', True)

    # Raw data with gaussian smoothing 50
    struct = {
        'features': [
            ('test/raw_mean_std_normalized', SimpleTransform()),
            ('test/raw_mean_std_normalized_smoothed_gaussian50', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'test/raw_normalized_gaussian50_dataset', True)

    struct = {
        'features': [
            ('test/detrend_gaussian10', SimpleTransform()),
            ('test/raw_mean_std_normalized_smoothed_gaussian50', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'test/detrend_gaussian10_with_smoothed_dataset', True)

    struct = {
        'features': [
            ('test/detrend_gaussian10', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'test/detrend_gaussian10_dataset', True)

    struct = {
        'features': [
            ('test/detrend_gaussian5', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'test/detrend_gaussian5_dataset', True)

    struct = {
        'features': [
            ('test/detrend_gaussian15', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'test/detrend_gaussian15_dataset', True)

    struct = {
        'features': [
            ('test/fft_smoothed_sigma10', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'test/fft_smoothed10_dataset', True)

    struct = {
        'features': [
             ('test/fft_smoothed_sigma20', SimpleTransform()),
        ],
    }
    generate_dataset(struct, 'test/fft_smoothed20_dataset', True)
