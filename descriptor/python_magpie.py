#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:

import joblib
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd

from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, AveragePooling1D,Convolution2D,AveragePooling2D,MaxPooling1D,MaxPooling2D
from keras.layers import concatenate
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import scipy.stats as stats
import numpy as np

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import sys
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from keras.callbacks import ModelCheckpoint, EarlyStopping
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
import tensorflow as tf
from tensorflow import keras

import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import numpy as np

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import sys


from sklearn import svm
from sklearn.model_selection import GridSearchCV


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold


import keras_tuner
from keras_tuner import Objective
import tensorflow as tf
from keras_tuner.tuners import BayesianOptimization
from keras.callbacks import ModelCheckpoint, EarlyStopping

# In[ ]:


X1_train=pd.read_csv(sys.argv[1])

X1_train = X1_train.to_numpy()

print(type(X1_train))



# In[6]:
y1_train=np.load(sys.argv[2], allow_pickle=True)

# In[ ]:
sample_size = X1_train.shape[0] # number of samples in train set
time_steps  = X1_train.shape[1] # number of features in train set
input_dimension = 1               # each feature is represented by 1 number

train_data_reshaped =X1_train.reshape(sample_size,time_steps,input_dimension)


# In[7]:


n_timesteps = train_data_reshaped.shape[1] #13
n_features  = train_data_reshaped.shape[2] #1

# In[9]:

def build_model11(hp):
    """Builds a convolutional model."""
    inputs = tf.keras.Input(shape=(132, 1))
    x = inputs
    for i in range(hp.Int("conv_layers", 1, 2, default=1)):
        x = tf.keras.layers.Convolution1D(
            filters=hp.Int("filters_" + str(i), 32, 256, step=32, default=64),
            kernel_size=hp.Int("kernel_size_" + str(i), 3, 9,step=1,default=5),
            activation="relu",
            padding="same",
        )(x)
        x = tf.keras.layers.ReLU()(x)
      
        x = tf.keras.layers.AveragePooling1D()(x)
        x=tf.keras.layers.Dropout(hp.Float('dropout_conv'+ str(i),0.3, 0.5, step=0.1, default=0.2))(x)


    x=tf.keras.layers.Flatten()(x)
    for i in range(hp.Int("dense_layers", 1, 3, default=1)):
        x = tf.keras.layers.Dense(
            units=hp.Int('dense_units'+str(i+10), min_value=32, max_value=512, step=32,default=32),
            activation='relu',
            
        )(x)
        x=tf.keras.layers.Dropout(hp.Float('dropout_dense_'+ str(i+10), 0.3, 0.5, step=0.1, default=0.3))(x)
        
        
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-01,1e-2, 1e-3,1e-4])),
             loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


# In[10]:

# In[ ]:
tuner_search=BayesianOptimization(build_model11,
                           objective=Objective('val_loss', direction="min"),
    max_trials=75,directory='gender_classifation', overwrite=True)


# In[ ]:


tuner_search.search(X1_train, y1_train, epochs=500,callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath='gender_prediction_best.keras',
            save_best_only='True',
            monitor='val_loss',patience=100
        )],validation_split=0.2, verbose=1)

model=tuner_search.get_best_models(num_models=1)[0]


# In[ ]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=250)
model.fit(X1_train, y1_train, epochs=1000, validation_split=0.2, initial_epoch=3,callbacks=[es])


print(model.summary())
model.save(sys.argv[3])

