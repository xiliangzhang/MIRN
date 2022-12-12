#!/usr/bin/env python
# encoding: utf-8

from tensorflow.python.keras.callbacks import Callback
class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses=[]
        self.acc=[]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
