#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
import pathlib
import csv


import warnings
warnings.filterwarnings('ignore')
import os
import gc
import logging
import argparse
from datetime import datetime
from collections import OrderedDict

# Disable TF warnings about speed up and future warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable warnings from h5py
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

# Audio processing and DL frameworks 
import h5py

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from gtanz import*

import keras
from keras import backend as K
from keras.models import load_model

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += " mfcc{}".format(i)
header += " label"
header = header.split()

if not os.path.exists('convertcsv.csv'):

  file = open('convertcsv.csv', 'w')
  with file:
      writer = csv.writer(file)
      writer.writerow(header)
  genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
  for g in genres:
      for filename in os.listdir("./genres/{}".format(g)):
          songname = "./genres/{}/{}".format(g,filename)
          y, sr = librosa.load(songname, mono=True, duration=30)
          
          chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
          rmse        = librosa.feature.rmse(y=y)
          spec_cent   = librosa.feature.spectral_centroid(y=y, sr=sr)
          spec_bw     = librosa.feature.spectral_bandwidth(y=y, sr=sr)
          rolloff     = librosa.feature.spectral_rolloff(y=y, sr=sr)
          zcr         = librosa.feature.zero_crossing_rate(y)
          mfcc        = librosa.feature.mfcc(y=y, sr=sr)
          flat        = librosa.feature.spectral_flatness(y)

          to_append = "{} {} {} {} {} {} {} {}".format(filename,np.mean(chroma_stft),np.mean(rmse),np.mean(spec_cent),np.mean(spec_bw),np.mean(rolloff),np.mean(zcr), np.mean(flat))
          for e in mfcc:
              to_append += " {}".format(np.mean(e))
          to_append += " {}".format(g)
          file = open('convertcsv.csv', 'a')
          with file:
              writer = csv.writer(file)
              writer.writerow(to_append.split())
            
            

data = pd.read_csv('convertcsv.csv')
data.head()

data.shape

data = pd.read_csv('convertcsv.csv')
data.head()

#data = data.drop(['filename'], axis=1)

from sklearn.preprocessing import LabelEncoder

num_genres = 10
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
Y = encoder.fit_transform(genre_list)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
exec_time = datetime.now().strftime('%Y%m%d%H%M%S')
#scaler = StandardScaler()
scaler = MinMaxScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.applications.vgg16 import VGG16

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
freezed_layers=5
input_shape = X_train.shape[1]
input_tensor = Input(shape=input_shape)
vgg16 = VGG16(include_top=False, weights='imagenet',input_tensor=input_tensor)
cnn = Sequential()
cnn.add(Flatten(input_shape=vgg16.output_shape[1:]))
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(num_genres, activation='softmax'))
model = Model(inputs=vgg16.input, outputs=cnn(vgg16.output))
for layer in model.layers[:freezed_layers]:
    layer.trainable = False
return model
cnn.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adam(),
metrics=['accuracy'])
   

    
hist = cnn.fit(X_train, Y_train,
                batch_size = 256,
                epochs = 50,
                verbose = 1,
                validation_data = (X_test, Y_test))

# Evaluate
score = cnn.evaluate(X_test, Y_test, verbose = 0)
print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))
# Plot graphs
save_history(hist, '../logs/{}/evaluate.png'.format(exec_time))
# Save the confusion Matrix
preds = np.argmax(cnn.predict(X_test), axis = 1)
y_orig = np.argmax(Y_test, axis = 1)
cm = confusion_matrix(preds, y_orig)

keys = OrderedDict(sorted(genres.items(), key=lambda t: t[1])).keys()
plot_confusion_matrix('../logs/{}/cm.png'.format(exec_time), cm, keys, normalize=True)
