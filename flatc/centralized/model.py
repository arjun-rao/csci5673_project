# coding: utf-8
import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import pandas as pd

from dataset import Dataset
from feature_extractor import FeatureExtractor
from nn import NN
from dataset_final import get_round_data

import keras
from keras import models
from keras.layers import Dropout, Dense
from keras import backend as K
from sklearn.metrics import classification_report
from IPython import embed

class Model:
    def __init__(self, x, y):
        # instantiate feature extractor
        self.fe = FeatureExtractor()
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        # instantiate keras model
        self.model = models.Sequential()
        


    def train(self, round_x, round_y):
        # TODO: Add round_x and round_y to self.X and self.Y
        # TODO: Fit Feature Extractor on X & Y
        self.X = self.fe.fit(round_x, round_y)
        self.Y = round_y
        input_shape = self.X.shape
        # Retrain the model with this data but don't reinitialize weights
        self.model.add(Dense(units=64, input_shape=input_shape, activation='relu'))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=1, activation="sigmoid"))

        optimizer = keras.optimizers.Adam(lr=1e-3)
        # Create callback for early stopping on validation loss. If the loss does
        # not decrease in two consecutive tries, stop training.
        callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2)]
        self.model.compile(optimizer=optimizer, loss="binary_crossentropy")
        history = self.model.fit(
            self.X,
            self.Y,
            epochs=10,
            callbacks=callbacks,
            validation_split=0.1,
            verbose=2,  # Logs once per epoch.
            batch_size=32)



    def predict(self, x):
        # Transform x into features using feature extractor
        #fit and then transform
        x_test = self.fe.fit(x)
        x_test = self.fe.transform(x_test)
        # predict using self.model.predict(x)
        y_pred = model.predict_classes(x_test)
        return y_pred
        
        

    def update_weights(weights):
        # TODO: Algorithm to average weights
        old_weights = self.model.get_weights()
        # TODO: Replace this line with actual for loop for averaging.
        averaged_weights = (old_weights + weights) / 2
        self.model.set_weights(averaged_weights)


if __name__ == "__main__":
    round1_x, round1_y = get_round_data('../data/0', 0)
    model = Model(round1_x, round1_y)
    round2_x, round2_y = get_round_data('../data/0', 1)
    model.train(round2_x, round2_y)





# def mlp_model(input_shape):
#     """Creates an instance of a multi-layer perceptron model.
#     # Returns
#         An MLP model instance.
#     """
    
#     # model.add(Dropout(rate=0.3, input_shape=input_shape))
#     model.add(Dense(units=64, input_shape=input_shape, activation='relu'))
#     # model.add(Dropout(rate=0.3))
#     model.add(Dense(units=64, activation='relu'))
#     # model.add(Dropout(rate=0.1))
#     model.add(Dense(units=1, activation="sigmoid"))
#     return model



# def recall_m(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall

# def precision_m(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

# # Create model instance.
# model = mlp_model(cx_train.shape[1:])
# optimizer = keras.optimizers.Adam(lr=1e-3)
# # Create callback for early stopping on validation loss. If the loss does
# # not decrease in two consecutive tries, stop training.
# callbacks = [keras.callbacks.EarlyStopping(
#         monitor='val_loss', patience=2)]
# model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[precision_m, recall_m])
# history = model.fit(
#           cx_train,
#           cy_train,
#           epochs=10,
#           callbacks=callbacks,
#           validation_data=(cx_test, cy_test),
#           verbose=2,  # Logs once per epoch.
#           batch_size=32)

# # Print results.
# history = history.history
# print('Validation Precision: {acc}, loss: {loss}'.format(
#         acc=history['val_precision_m'][-1], loss=history['val_loss'][-1]))


# y_pred = model.predict_classes(cx_test)
# y_pred = [item[0] for item in list(y_pred)]
# print(classification_report(cy_test, y_pred))


# # model = NN([806, 10, 1], keep_prob=0.2)
# # model.SGD_train(np.array(cx_train), cy_train, 10, 0.1, lam=0.001, test_x=np.array(cx_test), test_y=cy_train)

# #return trained model, feature_extractor