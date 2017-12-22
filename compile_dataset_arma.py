from utils.processing_helper import SimpleTransform, generate_dataset

if __name__ == '__main__':
    struct = {
        'features': [
            ('arma_20', SimpleTransform())
        ],
        'target': ('labels', SimpleTransform())
    }
    generate_dataset(struct, 'arma_20_dataset')
