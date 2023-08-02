import os
from sklearn import metrics
import keras
import tensorflow as tf
from glob import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import layers
import matplotlib.pyplot as plt


# Dataset Import
from zipfile import ZipFile

data_path = 'Leukemia.zip'

with ZipFile(data_path) as zip:
    zip.extractall('leukemia')


# Data Preparation
train_path = 'C-NMC_Leukemia/training_data'
test_path = 'C-NMC_Leukemia/testing_data'
val_path = 'C-NMC_Leukemia/validation_data'

classes = os.listdir(f'{train_path}')

IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 100
BATCH_SIZE = 61

X = []
Y = []

resize_list = [train_path, test_path]

for p in range(len(resize_list)):
    for i, name in enumerate(classes):
        images = glob(f'{resize_list[p]}/{name}/*.bmp')

        for image in images:
            img = cv2.imread(image)

            X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            Y.append(i)
X = np.asarray(X)
one_h_enc_Y = pd.get_dummies(Y).values

X_train, X_test, Y_train, Y_test = train_test_split(X, one_h_enc_Y, test_size= SPLIT, random_state= 42)


# Creating Model
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet", input_shape= IMG_SHAPE, pooling= 'max')


model = keras.Sequential([
    base_model,
    layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    layers.Dense(256, activation= 'relu'),
    layers.Dropout(rate= 0.45, seed= 123),
    layers.Dense(2, activation= 'softmax')
])

model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics= ['accuracy'])
print(model.summary())


# Callbacks
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('output/model_chechpoint.h5',
                             save_best_only= True,
                             verbose= 1,
                             save_weights_only= True,
                             monitor='val_accuracy')


# Model Training
history = model.fit(X_train, Y_train,
                    validation_data= (X_test, Y_test),
                    batch_size= BATCH_SIZE,
                    epochs= EPOCHS,
                    verbose= 1,
                    callbacks= checkpoint)
