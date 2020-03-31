# coding: utf-8
import sys
import os
import numpy as np
import pandas as pd
import json

from feature_extractor import FeatureExtractor
from dataset_final import get_round_data, get_data

import keras
from keras import models, initializers
from keras.layers import Dropout, Dense
from keras import backend as K
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from IPython import embed



from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


def mlp_model(input_shape):
    """Creates an instance of a multi-layer perceptron model.
    # Returns
        An MLP model instance.
    """
    model = models.Sequential()
    # model.add(Dropout(rate=0.3, input_shape=input_shape))
    model.add(Dense(units=64, input_shape=input_shape, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=42)))
    # model.add(Dropout(rate=0.3))
    model.add(Dense(units=64, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=42)))
    # model.add(Dropout(rate=0.1))
    model.add(Dense(units=1, activation="sigmoid", kernel_initializer=initializers.glorot_uniform(seed=42)))
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
            verbose=0,  # Logs once per epoch.
            batch_size=32)
        self.precision = self.history.history['val_precision_m'][-1]
        self.recall = self.history.history['val_recall_m'][-1]

    @staticmethod
    def compute_weighted_average(weights, num_samples):
        num_params = len(weights[0])
        average_weight = []
        total_N = sum(num_samples)
        # Number of params per client
        for i in range(num_params):
            result = np.zeros(weights[0][i].shape)
            # Number of clients
            for j in range(len(weights)):
                result += (float(num_samples[j])/total_N) * weights[j][i]
            average_weight.append(result)
        return average_weight

    def fine_tune(self):
         self.history = self.model.fit(
            self.X,
            self.Y,
            epochs=10,
            callbacks=self.callbacks,
            validation_split=0.1,
            verbose=0,  # Logs once per epoch.
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
            verbose=0,  # Logs once per epoch.
            batch_size=32)

    def predict_proba(self, x):
        # Transform x into features using feature extractor
        #fit and then transform
        x_test = self.fe.transform(x)
        # predict using self.model.predict(x)
        y_pred = self.model.predict_proba(x_test)
        return [item[0] for item in list(y_pred)]

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
        self.model.set_weights(weight_update)


    def update_weights_average(self, weight_update):
        new_weight = []
        old_weight = self.model.get_weights()
        for i in range(len(old_weight)):
            new_weight.append((old_weight[i] + weight_update[i])/2)
        self.model.set_weights(new_weight)




if __name__ == "__main__":
    test_x, test_y = get_data('./data/test.csv')

    def evaluate(model_i):
    # print(classification_report(test_y, model_i.predict(test_x)))
    # print('F1, ROC: {}, {}:'.format(f1_score(test_y, model_i.predict(test_x)),
    #     roc_auc_score(test_y, model_i.predict_proba(test_x))))
        return {
                    'f1': f1_score(test_y, model_i.predict(test_x)),
                    'roc': roc_auc_score(test_y, model_i.predict_proba(test_x))
        }

    def compare(_models):
        result = {}
        for i, model in enumerate(_models):
            result[i] = evaluate(model)
        return result
    # Test weighted average.
    _models = []
    n_clients = 3
    for i in range(n_clients):
        client_x, client_y = get_round_data('./data/'+str(i), 0)
        model = Model(client_x, client_y)
        _models.append(model)

    results = {}
    for i in range(1, 5):
        print('i: {}'.format(i))
        weights = [i.get_weights() for i in _models]
        n_samples = [i.x_text.shape[0] for i in _models]
        results[i] = {'before': compare(_models)}
        average_w = Model.compute_weighted_average(weights, n_samples)
        for j in range(n_clients):
            _models[j].update_weights_average(average_w)
            client_x, client_y = get_round_data('./data/'+str(j), i)
            model.train(client_x, client_y)
        results[i]['after'] = compare(_models)

    print(json.dumps(results, indent=2))


    # print(classification_report(test_y, model4.predict(test_x)))

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



