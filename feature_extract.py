from __future__ import division
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from scipy.signal import argrelextrema
from utils.processing_helper import save_features

dataset = np.load('features/detrend_gaussian10.npy')
def normalize(data):
    return (data-np.min(data))/(float(np.max(data))-float(np.min(data)))
     
def features(Data):
    x = Data

    Mean=np.mean(x,axis=1)
    Std_Dev=np.std(x,axis=1)
    Median=np.median(x,axis=1)

    T1=[]
    T2=[]
    T0=[]

    for j in range(x.shape[0]):
        t1=t2=t0=0
        for i in range(x.shape[1]):
            if Mean[j]-Std_Dev[j] > x[j][i] > Mean[j]-2*Std_Dev[j]:
                t1=t1+1
            elif x[j][i] < Mean[j]-2*Std_Dev[j]:
                t2=t2+1
            elif x[j][i] > 0:
                t0=t0+1

        T1.append(t1)
        T2.append(t2)
        T0.append(t0)
    T1=np.array(T1)
    T2=np.array(T2)
    T0=np.array(T0)

    Local_Max_Mean=[]
    Local_Max_Std=[]
    No_of_local_maxima=[]
    Local_Min_Mean=[]
    Local_Min_Std=[]
    No_of_local_minima=[]

    for i in range(x.shape[0]):
        Local_Max=x[i][argrelextrema(x[i], np.greater)[0]]
        No_of_local_maxima.append(len(Local_Max))
        Local_Max_Mean.append(np.mean(Local_Max))
        Local_Max_Std.append(np.std(Local_Max))

        Local_Min=x[i][argrelextrema(x[i], np.less)[0]]
        No_of_local_minima.append(len(Local_Min))
        Local_Min_Mean.append(np.mean(Local_Min))
        Local_Min_Std.append(np.std(Local_Min))

    Local_Min_Mean=np.array(Local_Min_Mean)    
    Local_Min_Std=np.array(Local_Min_Std)   
    No_of_local_minima=np.array(No_of_local_minima)
    
    Local_Max_Mean=np.array(Local_Max_Mean)    
    Local_Max_Std=np.array(Local_Max_Std)   
    No_of_local_maxima=np.array(No_of_local_maxima)

    Mean=normalize(Mean)
    Std_Dev=normalize(Std_Dev)
    Median=normalize(Median)
    T1=normalize(T1)
    T2=normalize(T2)
    T0=normalize(T0)
    Local_Max_Mean=normalize(Local_Max_Mean)
    Local_Max_Std=normalize(Local_Max_Std)
    No_of_local_maxima=normalize(No_of_local_maxima)
    Local_Min_Mean=normalize(Local_Min_Mean)
    Local_Min_Std=normalize(Local_Min_Std)
    No_of_local_minima=normalize(No_of_local_minima)

    features=np.column_stack  ((Mean,Std_Dev,Median,T0,T1,T2,Local_Max_Mean,Local_Max_Std,No_of_local_maxima,Local_Min_Mean,Local_Min_Std,No_of_local_minima))
    return features
# Create PolynomialFeatures object with interaction_only set to True
interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

features_ = features(dataset)
features_dataset=interaction.fit_transform(features_)
save_features(features_dataset,'features_paper')
