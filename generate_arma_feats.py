import pandas as pd
import numpy as np
import pyflux as pf
from config import raw_data_filename, FEATURES_PATH
from utils.processing_helper import save_features

def generate_arma_feats(x_dataset):
     arma_20=[]
     for i in range(3960):
        arma = pf.ARIMA(data=x_dataset[i], ar=20, ma=20, family=pf.Normal()).fit('MLE')
        data = []
        for i in range(len(arma.z.z_list)):
            data.append(np.round(arma.z.z_list[i].prior.transform(arma.results.x[i]),4))
        arma_20.append(data)

     save_features(arma_20, 'arma_20')

if __name__ == '__main__':
    # TODO Add test file code
    dataset = pd.read_csv(raw_data_filename)

    print('Extracting time series features...')
    x_dataset = dataset.iloc[:, 1:]
    x_dataset=np.array(x_dataset)
    generate_arma_feats(x_dataset)
