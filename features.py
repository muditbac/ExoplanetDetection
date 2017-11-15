from __future__ import division
import numpy as np
import pandas as pd
import keras
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import argrelextrema

raw_data = np.loadtxt('/home/saranshk2/Downloads/ExoTrain.csv', skiprows=1, delimiter=',')

x = raw_data[:, 1:]
y = raw_data[:, 0, np.newaxis] - 1
x_1=np.transpose(x)
X_std = (x_1 - x.min(axis=1)) / (x.max(axis=1) - x.min(axis=1))
print X_std.shape
x_scaled=np.transpose(X_std)


x3=[]
x2=[]
x4=[]

from sklearn import preprocessing
x1=uniform_filter1d(x_scaled, axis=1, size=15)

plt.plot(x1[3])
plt.show()
for i in range(x1.shape[0]):
    s=(x_scaled[i]-x1[i])
    x3.append(s)
x3=np.array(x3)
plt.axhline(y=np.mean(x3[3])-np.std(x3[3]))
plt.axhline(y=np.mean(x3[3])-2*np.std(x3[3]))
plt.plot(x3[3])
plt.show()

Mean=np.mean(x3,axis=1)
Std_Dev=np.std(x3,axis=1)
Median=np.median(x3,axis=1)
T1=[]
T2=[]
T0=[]
for j in range(3960):
    t1=t2=t0=0
    for i in range(3197):
        
        if Mean[j]-Std_Dev[j] > x3[j][i] > Mean[j]-2*Std_Dev[j]:
            t1=t1+1
        elif x3[1][i] < Mean[0]-2*Std_Dev[0]:
            t2=t1+1
        elif x3[1][i] > 0:
            t0=t0+1
    T1.append(t1)
    T2.append(t2)
    T0.append(t0)
T1=np.array(T1)
T2=np.array(T2)
T0=np.array(T0)
print T1.shape  
Local_Max_Mean=[]
Local_Max_Std=[]
No_of_local_minina=[]
for i in range(3960):
    Local_Max=x3[i][argrelextrema(x3[i], np.greater)[0]]
    No_of_local_minina.append(len(Local_Max))
    Local_Max_Mean.append(np.mean(Local_Max))
    Local_Max_Std.append(np.std(Local_Max))
Local_Max_Mean=np.array(Local_Max_Mean)    
Local_Max_Std=np.array(Local_Max_Std)   
No_of_local_minina=np.array(No_of_local_minina)
print No_of_local_minina
f1 = "/home/saranshk2/Documents/feature.csv"
outfile = open(f1, "a")
Mean=(Mean-np.min(Mean))/(float(np.max(Mean))-float(np.min(Mean)))
Std_Dev=(Std_Dev-np.min(Std_Dev))/(float(np.max(Std_Dev))-float(np.min(Std_Dev)))
Median=(Median-np.min(Median))/(float(np.max(Median))-float(np.min(Median)))
T1=(T1-np.min(T1))/(float(np.max(T1))-float(np.min(T1)))
T2=(T2-np.min(T2))/(float(np.max(T2))-float(np.min(T2)))
T0=(T0-np.min(T0))/(float(np.max(T0))-float(np.min(T0)))
Local_Max_Mean=(Local_Max_Mean-np.min(Local_Max_Mean))/(float(np.max(Local_Max_Mean))-float(np.min(Local_Max_Mean)))
Local_Max_Std=(Local_Max_Std-np.min(Local_Max_Std))/(float(np.max(Local_Max_Std))-float(np.min(Local_Max_Std)))
No_of_local_minina=(No_of_local_minina-np.min(No_of_local_minina))/(float(np.max(No_of_local_minina))-float(np.min(No_of_local_minina)))

for i in range(3960):
    a=Mean[i]
    b=Std_Dev[i]
    c=Median[i]
    d=T1[i]
    e=T2[i]
    f=T0[i]
    g=Local_Max_Mean[i]
    h=Local_Max_Std[i]
    j=No_of_local_minina[i]
    
    outfile.write(str(a)+','+str(b)+','+str(c)+','+str(d)+','+str(e)+','+str(f)+','+str(g)+','+str(h)+','+str(j)+','+str(a*b)+','+str(a*c)+','+str(a*d)+','+str(a*e)+','+str(a*f)+','+str(a*g)+','+str(a*h)+','+str(a*j)+','+str(b*c)+','+str(b*d)+','+str(b*e)+','+str(b*f)+','+str(b*g)+','+str(b*h)+','+str(b*j)+','+str(c*d)+','+str(c*e)+','+str(c*f)+','+str(c*g)+','+str(c*h)+','+str(c*j)+','+str(d*e)+','+str(d*f)+','+str(d*g)+','+str(d*h)+','+str(d*j)+','+str(e*f)+','+str(e*g)+','+str(e*h)+','+str(e*j)+','+str(f*g)+','+str(f*h)+','+str(f*j)+','+str(g*h)+','+str(g*j)+','+'\n')
"""for i in range(3960):
    outfile.write(str(Mean[i])+','+str(Std_Dev[i])+','+str(Median[i])+','+str(T2[i])+','+str(T1[i])+','+str(T0[i])+','+str(Local_Max_Mean[i])+','+str(Local_Max_Std[i]+'\n')) """
