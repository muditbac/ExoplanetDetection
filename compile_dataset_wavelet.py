from utils.processing_helper import SimpleTransform, generate_dataset

if __name__ == '__main__':
    struct = {
        'features': [
            ('wavelet_db2_a', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'wavelet_db2_a_dataset')

    struct = {
        'features': [
            ('wavelet_db2_b', SimpleTransform()),
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'wavelet_db2_b_dataset')
