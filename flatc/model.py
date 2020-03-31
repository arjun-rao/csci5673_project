# coding: utf-8
import sys
import os
import numpy as np
import pandas as pd

from feature_extractor import FeatureExtractor
from dataset_final import get_round_data, get_data

import keras
from keras import models
from keras.layers import Dropout, Dense
from keras import backend as K
from sklearn.metrics import classification_report
from IPython import embed


def mlp_model(input_shape):
    """Creates an instance of a multi-layer perceptron model.
    # Returns
        An MLP model instance.
    """
    model = models.Sequential()
    # model.add(Dropout(rate=0.3, input_shape=input_shape))
    model.add(Dense(units=64, input_shape=input_shape, activation='relu'))
    # model.add(Dropout(rate=0.3))
    model.add(Dense(units=64, activation='relu'))
    # model.add(Dropout(rate=0.1))
    model.add(Dense(units=1, activation="sigmoid"))
    return model

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class Model:
    def __init__(self, x, y):
        # instantiate feature extractor
        self.fe = FeatureExtractor()
        self.x_text = x
        self.X = self.fe.fit(x, y)
        self.Y = y
        input_shape = self.X.shape[1:]
        # instantiate keras model
        self.model = mlp_model(input_shape)
        self.optimizer = keras.optimizers.Adam(lr=1e-3)
        # Create callback for early stopping on validation loss. If the loss does
        # not decrease in two consecutive tries, stop training.
        self.callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2)]
        self.model.compile(optimizer=self.optimizer, loss="binary_crossentropy",  metrics=[precision_m, recall_m])
        self.history = self.model.fit(
            self.X,
            self.Y,
            epochs=10,
            callbacks=self.callbacks,
            validation_split=0.1,
            verbose=2,  # Logs once per epoch.
            batch_size=32)
        self.precision = self.history.history['val_precision_m'][-1]
        self.recall = self.history.history['val_recall_m'][-1]

    @staticmethod
    def compute_weighted_average(weights, num_samples):
        pass

    def fine_tune(self):
         self.history = self.model.fit(
            self.X,
            self.Y,
            epochs=10,
            callbacks=self.callbacks,
            validation_split=0.1,
            verbose=2,  # Logs once per epoch.
            batch_size=32)

    def train(self, round_x, round_y):
        # TODO: Add round_x and round_y to self.X and self.Y
        self.x_text = pd.concat([self.x_text, round_x])
        self.Y = pd.concat([self.Y, round_y])
        self.X = self.fe.fit(self.x_text, self.Y)
        # TODO: Fit Feature Extractor on X & Y
        self.history = self.model.fit(
            self.X,
            self.Y,
            epochs=10,
            callbacks=self.callbacks,
            validation_split=0.1,
            verbose=2,  # Logs once per epoch.
            batch_size=32)

    def predict(self, x):
        # Transform x into features using feature extractor
        #fit and then transform
        x_test = self.fe.transform(x)
        # predict using self.model.predict(x)
        y_pred = self.model.predict_classes(x_test)
        return [item[0] for item in list(y_pred)]

    def get_weights(self):
        return self.model.get_weights()


    def update_weights(self, weight_update):
        # TODO: Algorithm to average weights
        old_weights = self.model.get_weights()
        new_weights = []
        for i in range(len(old_weights)):
            new_weights.append((old_weights[i] + weight_update[i])/2)
        self.model.set_weights(new_weights)


if __name__ == "__main__":
    test_x, test_y = get_data('./data/test.csv')

    # Test weighted average.
    client0_x, client0_y = get_round_data('./data/0', 0)
    model0 = Model(client0_x, client0_y)
    print(classification_report(test_y, model0.predict(test_x)))

    client1_x, client1_y = get_round_data('./data/1', 0)
    model1 = Model(client1_x, client1_y)
    print(classification_report(test_y, model1.predict(test_x)))

    client2_x, client2_y = get_round_data('./data/2', 0)
    model2 = Model(client2_x, client2_y)
    print(classification_report(test_y, model2.predict(test_x)))

    client3_x, client3_y = get_round_data('./data/3', 0)
    model3 = Model(client3_x, client3_y)
    print(classification_report(test_y, model3.predict(test_x)))

    client4_x, client4_y = get_round_data('./data/4', 0)
    model4 = Model(client4_x, client4_y)
    print(classification_report(test_y, model4.predict(test_x)))

    # Test sequential rounds
    # round1_x, round1_y = get_round_data('./data/0', 0)
    # model = Model(round1_x, round1_y)
    # test_x, test_y = get_data('./data/test.csv')
    # y_pred = model.predict(test_x)
    # print(classification_report(test_y, y_pred))
    # round2_x, round2_y = get_round_data('./data/0', 1)
    # model.train(round2_x, round2_y)
    # y_pred = model.predict(test_x)
    # print(classification_report(test_y, y_pred))






