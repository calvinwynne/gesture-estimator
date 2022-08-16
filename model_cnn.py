import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard


class CNN:
    
    def __init__(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(216, 748, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(11, activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        print(model.summary())
        self.model = model
        self.actions = ['handshake', 'waving', 'yawning', 'walking', 'bowing', 'punching', 'standing', 'sitting', 'touchinghead', 'defending', 'reachingup']
        
    def prepare_dataset(self):
        dataset = flatten_dataset(master_dataset, level=1)
        labels  = encode_labels(master_labels)
        dataset = shuffle_dataset(dataset, labels)
        self.train, self.train_label, self.test, self.test_label = train_test_split(dataset, train_test_ratio = 0.8, flatten_label=True)   
        
    def train(self):
        model.fit(train, train_label, epochs=50, callbacks=[tb_callback])

    def test_accuracy(self):
        predicted = self.model.predict(self.test)
        counter = 0
        for i in range(predicted.shape[0]):
            if self.actions[np.argmax(predicted[i])] == self.actions[np.argmax(self.test_label[i])]:
                counter +=1     
        print("Accuracy over test data:", counter/predicted.shape[0])
        
    def train_accuracy(self):
        predicted = self.model.predict(self.train)
        counter = 0
        for i in range(predicted.shape[0]):
            if self.actions[np.argmax(predicted[i])] == self.actions[np.argmax(self.train_label[i])]:
                counter +=1     
        print("Accuracy over train data:", counter/predicted.shape[0])
        
    