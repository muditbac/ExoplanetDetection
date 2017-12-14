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
