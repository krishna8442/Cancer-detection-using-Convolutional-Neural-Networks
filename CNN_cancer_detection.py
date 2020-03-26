"""
Created on Sat Sep 21 12:12:29 2019

@author: krish
"""

import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier=Sequential()
classifier.add(Conv2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('/Users/krish/Desktop/cancer_cnn/training',
                       target_size=(32, 32),
                       batch_size=10,
                       class_mode='binary')
test_set = test_datagen.flow_from_directory('/Users/krish/Desktop/cancer_cnn/test',
                  target_size=(32, 32),
                  batch_size=32,
                  class_mode='binary')
classifier.fit_generator(training_set,
                         steps_per_epoch=1000,
                         epochs=5,
                         validation_data= test_set, 
                         validation_steps=400)

classifier.evaluate_generator(test_set,steps_per_epoch=1000,epochs=5, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

