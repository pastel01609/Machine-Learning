# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 00:21:47 2020

@author: AAAAAAAAAAAAAAAAAA
"""


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from PIL import Image
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


       
    
model = Sequential()


#Conv2d > MaxPooling2D, Dropout, Flatten, Dense, Dropout, sigmoid

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
model.add(Conv2D(32, (3,3), activation = "relu", input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "sigmoid"))




model.compile(loss = 'binary_crossentropy',optimizer = 'rmsprop', metrics = ['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("C:/Users/silva\Documents/programming/datasci/Neural-Networks/chest_xray/train",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = test_datagen.flow_from_directory("C:/Users/silva\Documents/programming/datasci/Neural-Networks/chest_xray/val",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory("C:/Users/silva\Documents/programming/datasci/Neural-Networks/chest_xray/test",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

model_gen = model.fit_generator(training_set,
                            steps_per_epoch = 163,
                            epochs = 10,
                            validation_data = validation_set,
                            validation_steps = 624/32)


test_accu = model.evaluate_generator(test_set,steps=624)
print('The testing accuracy is :',test_accu[1]*100, '%')
