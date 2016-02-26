#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import dataset, models
from keras.callbacks import ModelCheckpoint, ProgbarLogger

basepath = os.path.dirname(__file__)

print "load data",
X, y = dataset.load_as_nparray()
X_train, y_train = X[:800,:,:], y[:800,:]
X_test, y_test = X[800:,:,:], y[800:,:]

print "compile model",
model = models.VGG()

checkpoint = ModelCheckpoint(filepath=os.path.join(basepath, 'checkpoint.hdf5'), verbose=1, save_best_only=True)
model.fit(X_train, y_train, batch_size=100, nb_epoch=20, verbose=1, validation_data=(X_test, y_test), callbacks=[checkpoint])