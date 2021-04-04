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

import keras

import warnings
warnings.filterwarnings('ignore')

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

#          chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#          rmse        = librosa.feature.rmse(y=y)
#          spec_cent   = librosa.feature.spectral_centroid(y=y, sr=sr)
#          spec_bw     = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#          rolloff     = librosa.feature.spectral_rolloff(y=y, sr=sr)
          zcr         = librosa.feature.zero_crossing_rate(y)
          mfcc        = librosa.feature.mfcc(y=y, sr=sr)
#          flat        = librosa.feature.spectral_flatness(y)
          mel         = librosa.feature.melspectrogram(y=y, sr=sr)

#          to_append = "{} {} {} {} {} {} {} {} {}".format(filename,np.mean(chroma_stft),np.mean(rmse),np.mean(spec_cent),np.mean(spec_bw),np.mean(rolloff),np.mean(zcr), np.mean(flat),np.mean(mfcc))
#          to_append = "{} {} {} {} {} {} {} {} {} {}".format(filename,np.mean(mfcc),np.var(mfcc),np.std(mfcc),np.mean(spec_cent),np.mean(spec_bw),np.mean(rolloff),np.mean(zcr), np.mean(flat),np.mean(chroma_stft))
          to_append = "{} {} {} {} {} {}".format(filename,np.mean(mfcc),np.var(mfcc),np.std(mfcc),np.mean(mel),np.mean(zcr))
          for e in mfcc:
              to_append += " {}".format(np.mean(e))
          to_append += " {}".format(g)
          file = open('convertcsv.csv', 'a')
          with file:
              writer = csv.writer(file)
              writer.writerow(to_append.split())
            
            

data = pd.read_csv('convertcsv.csv')
data.head()

from sklearn.preprocessing import LabelEncoder

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
Y = encoder.fit_transform(genre_list)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#scaler = StandardScaler()
scaler = MinMaxScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l1, l2

model = Sequential()

model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()
optim = keras.optimizers.sgd(lr=0.001)
#optim = keras.optimizers.rmsprop(lr=0.001)
#optim = keras.optimizers.adam(lr=0.001)
model.compile(optimizer=optim,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=2, mode='auto')

hist = model.fit(X_train,
                 Y_train,
                 epochs=450,
                 batch_size=25,
                 verbose = 2,
                 validation_data=(X_test,Y_test),
                  callbacks = [early_stop],
                 shuffle=True
                )
plt.figure(0)
plt.plot(hist.history['acc'],'r')
plt.plot(hist.history['val_acc'],'g')
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])
 
plt.figure(1)
plt.plot(hist.history['loss'],'r')
plt.plot(hist.history['val_loss'],'g')
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])
 
plt.show()
predictions = model.predict(X_test)
predictions.shape

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.DataFrame()
df["predictions"] = np.argmax(predictions,axis=1)
df["actual labels"] = Y_test
confu = confusion_matrix(np.argmax(predictions,axis=1),Y_test)
confu
acur = accuracy_score(np.argmax(predictions,axis=1),Y_test)
print(confu)
print("accuracy = ",acur*100)