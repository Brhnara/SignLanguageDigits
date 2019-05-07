# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:17:02 2019

@author: LENOVO
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# load data set
x = np.load('datas.npy')
y = np.load('labels.npy')

img_size = 100
# plt.subplot(1,2,1)
# plt.imshow(x[700].reshape(img_size,img_size))
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.imshow(x[1500].reshape(img_size,img_size))
# plt.axis('off')


# As you can see, y (labels) are already one hot encoded
print(y.max())
print(y.min())
print(y.shape)

# And x (features) are already scaled between 0 and 1
# print(x.max())
# print(x.min())
# print(x.shape)

# Now,lets create x_train, y_train, x_test, y_test arrays
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
#reshape
# x_train = x_train.reshape(-1,img_size,img_size,1)
# x_test = x_test.reshape(-1,img_size,img_size,1)
#print x_train and y_train shape
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# input()

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential # to create a cnn model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape=(100,100,1)))
model.add(MaxPool2D(pool_size = (2,2)))
# model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
# model.add(Dropout(0.25))


model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
# model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))


# fully connected
model.add(Flatten())

model.add(Dense(1024, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))


model.summary()


# Define the optimizer
optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
# Compile the model
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train,y_train, callbacks=[mc], epochs=25,validation_data=(x_test,y_test))



#visualization




model.save('model.h5')
