# coding: utf-8
import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np

from flatc.dataset import Dataset
from flatc.feature_extractor import FeatureExtractor
from flatc.nn import NN

from IPython import embed

d = Dataset("../data/spam.csv")
d.train_test_split(0.2)
_ = d.client_split(3)
_ = d.train_step_split(3)
cX = d.train_step_splits_X
cy = d.train_step_splits_y

fe = FeatureExtractor()
cx_train = fe.fit(cX[0][0], cy[0][0])
cy_train = cy[0][0]
# cx_train = fe.fit(d.X_train, d.y_train)
# cy_train = d.y_train
cx_test = fe.transform(d.X_test)
cy_test = d.y_test

import keras
from keras import models
from keras.layers import Dropout, Dense

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

from keras import backend as K

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

# Create model instance.
model = mlp_model(cx_train.shape[1:])
optimizer = keras.optimizers.Adam(lr=1e-3)
# Create callback for early stopping on validation loss. If the loss does
# not decrease in two consecutive tries, stop training.
callbacks = [keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[precision_m, recall_m])
history = model.fit(
          cx_train,
          cy_train,
          epochs=10,
          callbacks=callbacks,
          validation_data=(cx_test, cy_test),
          verbose=2,  # Logs once per epoch.
          batch_size=32)

# Print results.
history = history.history
print('Validation Precision: {acc}, loss: {loss}'.format(
        acc=history['val_precision_m'][-1], loss=history['val_loss'][-1]))

from sklearn.metrics import classification_report
y_pred = model.predict_classes(cx_test)
y_pred = [item[0] for item in list(y_pred)]
print(classification_report(cy_test, y_pred))


# model = NN([806, 10, 1], keep_prob=0.2)
# model.SGD_train(np.array(cx_train), cy_train, 10, 0.1, lam=0.001, test_x=np.array(cx_test), test_y=cy_train)
