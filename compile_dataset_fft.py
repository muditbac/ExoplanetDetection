from utils.processing_helper import SimpleTransform, generate_dataset

if __name__ == '__main__':
    struct = {
        'features': [
             ('fft_smoothed_sigma10', SimpleTransform()),
            # ('fft_half_normalized', SimpleTransform()),
            #('fft_normalized', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }

    generate_dataset(struct, 'fft_smoothed10_dataset')