from utils.processing_helper import SimpleTransform, generate_dataset

if __name__ == '__main__':
    struct = {
        'features': [
            ('fft_smoothed_sigma10', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'fft_smoothed10_dataset')

    struct = {
        'features': [
            ('fft_smoothed_sigma20', SimpleTransform()),
            # ('fft_half_normalized', SimpleTransform()),
            # ('fft_normalized', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }

    generate_dataset(struct, 'fft_smoothed20_dataset')

    struct = {
        'features': [
            ('fft_smoothed_median81', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'fft_smoothed_median81_dataset')
