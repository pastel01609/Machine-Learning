# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 00:58:23 2020

@author: AAAAAAAAAAAAAAAAAA
"""


from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

(mnist_train_images, mnist_train_labels),(mnist_test_images,mnist_test_labels) = mnist.load_data()


#convert images to floating point number then dividing by 255 to turn into 1 or 0
train_images = mnist_train_images.reshape(60000, 784)
test_images = mnist_test_images.reshape(10000, 784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255


#encode labels
train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(256,activation = "relu"))
model.add(Dense(10,activation = "softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    batch_size=100,
                    epochs=100,
                    verbose=2,
                    validation_data=(test_images, test_labels))

score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])