from utils.processing_helper import SimpleTransform, generate_dataset

if __name__ == '__main__':
    struct = {
        'features': [
            ('raw_ar_coeff100', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_ar_coeff100_dataset')

    struct = {
        'features': [
            ('raw_ar_coeff500', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'raw_ar_coeff500_dataset')

    struct = {
        'features': [
            ('fft_smoothed_sigma10', SimpleTransform()),
            ('raw_ar_coeff100', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'fft_smoothed10_ar100_dataset')