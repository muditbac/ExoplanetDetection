import pandas as pd
import numpy as np
import pyflux as pf
from config import raw_data_filename, FEATURES_PATH
from utils.processing_helper import save_features

def generate_arma_feats(x_dataset):
     arma_20=[]
     for i in range(x_dataset.shape[0]):
        arma = pf.ARIMA(data=x_dataset[i], ar=20, ma=20, family=pf.Normal()).fit('MLE')
        data = []
        for i in range(len(arma.z.z_list)):
            data.append(np.round(arma.z.z_list[i].prior.transform(arma.results.x[i]),4))
        arma_20.append(data)

     return np.array(arma_20)

