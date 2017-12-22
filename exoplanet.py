import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.ndimage.filters import uniform_filter1d
from sklearn.cross_validation import train_test_split

"""dataset = pd.read_csv('/home/saranshk2/Downloads/ExoTrain.csv')
x=dataset.iloc[1:,1:].values
y=dataset.iloc[1:,0].values
print len(y)
print y
for i in range(3959):
    if y[i]==2:
        y[i]=1
    else:
        y[i]=0
x=np.array(x)
y=np.array(y)
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train=x[16:]
y_train=y[16:]
x_test=x[:16,:]
y_test=y[0:16]

print len(x_test[0])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#x_train = PCA(n_components=50).fit_transform(x_train)
#x_test=PCA(n_components=50).fit_transform(x_test)
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(units=15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 3197))
classifier.add(Dense(units=15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units=15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 7)
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print cm"""

"""x=x[0]
y=range(1022)
y=np.array(y)
plt.plot(x)
plt.show()"""
from sklearn.decomposition import PCA
dataset = pd.read_csv('/home/saranshk2/Downloads/ExoTrain.csv')
x_train=dataset.iloc[1:,1:].values
y_train=dataset.iloc[1:,0].values

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
sc = preprocessing.MinMaxScaler()
x_train = sc.fit_transform(x_train)

for i in range(3959):
    if y_train[i]==2:
        y_train[i]=1
    else:
        y_train[i]=0

pca=PCA(n_components=2)
x1_train=pca.fit_transform(x_train)

def plot_resampling(ax, X, y, title):
    c0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0",color='g', alpha=0.5)
    c1 = ax.scatter(X[y == 1, 0], X[y == 1, 1],label="Class #1", alpha=0.5)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-1, 1])

    return c0,c1
kind = ['regular', 'borderline1', 'borderline2', 'svm']
sm = [SMOTE(kind=k) for k in kind]
X_resampled = []
y_resampled = []
X_res_vis = []
for method in sm:
    X_res, y_res = method.fit_sample(x_train, y_train)
    X_resampled.append(X_res)
    y_resampled.append(y_res)
    X_res_vis.append(pca.transform(X_res))

f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
# Remove axis for second plot
ax2.axis('off')
ax_res = [ax3, ax4, ax5, ax6]

c0, c1 = plot_resampling(ax1, x1_train, y_train, 'Original set')
for i in range(len(kind)):
    plot_resampling(ax_res[i], X_res_vis[i], y_resampled[i],
                    'SMOTE {}'.format(kind[i]))

ax2.legend((c0, c1), ('Class #0', 'Class #1'), loc='center',
           ncol=1, labelspacing=0.)
plt.tight_layout()
plt.show()

"""
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(units=15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 3197))
classifier.add(Dense(units=15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units=15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_resampled[0], y_resampled[0], batch_size = 10, epochs = 50)"""
