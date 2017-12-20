import cPickle
import pickle
import numpy as np
import json
from config import FEATURES_PATH, DATASETS_PATH, FOLDS_FILENAME, MODELFILE_PATH
import os
import pickle as pkl
from keras.models import model_from_json
from utils.processing_helper import make_dir_if_not_exists


def save_model(model, model_filename, cnn=False):
    """
    Saves the model
    :param model: model object
    :param model_filename: File name of the model
    """
    print 'Saving the model...'
    if not cnn:
        model_filename = os.path.join(MODELFILE_PATH, '%s.model' % model_filename)
        make_dir_if_not_exists(os.path.dirname(model_filename))
        with open(model_filename, 'wb') as fp:
            pickle.dump(model, fp)
    else:
        json_model = model.model.to_json()
        model_filename_archi = os.path.join(MODELFILE_PATH, '%s_archi.model' % model_filename)
        make_dir_if_not_exists(os.path.dirname(model_filename_archi))
        model_filename_weights = os.path.join(MODELFILE_PATH, '%s_weights.model' % model_filename)
        make_dir_if_not_exists(os.path.dirname(model_filename_weights))
        # Save the architecture
        with open(model_filename_archi, 'w') as f:
            f.write(json_model)
        # Save the weights
        model.model.save_weights(model_filename_weights, overwrite=True)

        model.model = None
        model_filename = os.path.join(MODELFILE_PATH, '%s.model' % model_filename)
        with open(model_filename, 'wb') as fp:
            cPickle.dump(model, fp)


def load_model(dataset_name, model_file_name, cnn=False):
    """
    Loads a model
    :param model_file_name: Name of the model to load
    """
    if not cnn:
        with open(os.path.join(MODELFILE_PATH, dataset_name+'_'+model_file_name+'.model'), 'rb') as fp:
            return cPickle.load(fp)
    else:
        archi_file = os.path.join(MODELFILE_PATH, dataset_name+'_'+model_file_name+'_archi'+'.model')
        weights_file = os.path.join(MODELFILE_PATH, dataset_name+'_'+model_file_name+'_weights'+'.model')
        model = model_from_json(open(archi_file).read())
        model.load_weights(weights_file)

        model_filename = os.path.join(MODELFILE_PATH, '%s_%s.model' % (dataset_name, model_file_name))
        with open(model_filename, 'rb') as fp:
            model_wrapper = cPickle.load(fp)
        model_wrapper.model = model
        return model_wrapper