#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os

import numpy
from keras.callbacks import ModelCheckpoint

import models
from precise_location.use_keras import dataset

basepath = os.path.dirname(os.path.abspath(__file__))

X, y = dataset.load_as_nparray()
X_train, y_train = X[:800, :, :], y[:800, :]
X_test, y_test = X[800:, :, :], y[800:, :]

model = models.ConvolutionalNetwork()

checkpoint = ModelCheckpoint(filepath=os.path.join(basepath, 'checkpoint.hdf5'), verbose=1, save_best_only=True)
history = model.fit(X_train, y_train, batch_size=50, nb_epoch=50, verbose=1, validation_data=(X_test, y_test), callbacks=[checkpoint])

y_pred = model.predict(X)
error = numpy.abs(y - y_pred).mean(axis=1)

print '全部平均误差: %.3f' % error.mean()
print '测试平均误差: %.3f' % error[-200:].mean()
print '最大误差: %.3f, 位于%d' % (error.max(), error.argmax())
print '最大误差样本数据结果对比: '
print '-- 实际值' + str(y[error.argmax()])
print '-- 预测值' + str(y[error.argmax()])