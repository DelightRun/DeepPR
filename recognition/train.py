import os
from six.moves import cPickle
from keras.utils import np_utils
import numpy as np
import cv2

import models

nb_epoch = 20
batch_size = 128

with open(os.path.join('data', 'chinese_data.pkl'), 'rb') as f:
    chinese_x,chinese_y = cPickle.load(f, encoding='bytes')
with open(os.path.join('data', 'alnum_data.pkl'), 'rb') as f:
    alnum_x,alnum_y = cPickle.load(f, encoding='bytes')

chinese_y = np_utils.to_categorical(chinese_y, 31)
alnum_y = np_utils.to_categorical(alnum_y, 34)

model = models.create_chinese_model()

model.fit(chinese_x, chinese_y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.2)
model.save_weights(os.path.join('trained_models', 'chinese_weights.h5'))

model = models.create_alnum_model()

model.fit(alnum_x, alnum_y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.2)
model.save_weights(os.path.join('trained_models', 'alnum_weights.h5'))
