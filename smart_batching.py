import numpy as np
import pandas as pd
import keras
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.ndimage.filters import uniform_filter1d
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA

raw_data = np.loadtxt('/home/saranshk2/Downloads/ExoTrain.csv', skiprows=1, delimiter=',')
x_train = raw_data[:, 1:]
y_train = raw_data[:, 0, np.newaxis] - 1
x_test = np.vstack((x_train[:12, :], x_train[3500:, :]))
x_train = x_train[12:3500, :]
y_test = np.vstack((y_train[:12, :], y_train[3500:, :]))
y_train = y_train[12:3500, :]

del raw_data
x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / 
           np.std(x_train, axis=1).reshape(-1,1))
x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / 
           np.std(x_test, axis=1).reshape(-1,1))

x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=200)], axis=2)
x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)


def batch_generator(x_train, y_train, batch_size=32):
    
    
    half_batch = batch_size // 2
    x_batch = np.empty((batch_size, x_train.shape[1],x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    
    yes_idx = np.where(y_train[:,0] == 1.)[0]
    non_idx = np.where(y_train[:,0] == 0.)[0]
    
    while True:
        np.random.shuffle(yes_idx)
        np.random.shuffle(non_idx)
    
        x_batch[:half_batch] = x_train[yes_idx[:half_batch]]
        x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
        y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]
    
        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
     
        yield x_batch[:,:,1], y_batch

from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 3197))
classifier.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units=100, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
hist = classifier.fit_generator(batch_generator(x_train, y_train, 32), 
                           validation_data=(x_test[:,:,1], y_test), 
                           verbose=1, epochs=50,
                           steps_per_epoch=x_train.shape[0]//32)
plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()





