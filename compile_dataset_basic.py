from utils.processing_helper import SimpleTransform, generate_dataset

if __name__ == '__main__':
    struct = {
        'features': [
            ('raw_mean_std_normalized', SimpleTransform()),
            ('raw_mean_std_normalized_smoothed_uniform200', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_normalized_smoothed_dataset')

    # Raw data with gaussian smoothing 50
    struct = {
        'features': [
            ('raw_mean_std_normalized', SimpleTransform()),
            ('raw_mean_std_normalized_smoothed_gaussian50', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_normalized_gaussian50_dataset')

    struct = {
        'features': [
            ('detrend_gaussian10', SimpleTransform()),
            ('raw_mean_std_normalized_smoothed_gaussian50', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'detrend_gaussian10_with_smoothed_dataset')

    struct = {
        'features': [
            ('detrend_gaussian10', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'detrend_gaussian10_dataset')

    struct = {
        'features': [
            ('detrend_gaussian5', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'detrend_gaussian5_dataset')

    struct = {
        'features': [
            ('detrend_gaussian15', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'detrend_gaussian15_dataset')
