import os
import numpy
from keras.utils import np_utils

from . import models

basedir = os.path.dirname(os.path.realpath(__file__))

# chinese_model = models.create_chinese_model()
# chinese_model.load_weights(os.path.join(basedir, 'trained_models', 'chinese_weights.h5'))
# chinese_labels = [line.strip() for line in open(os.path.join(basedir, 'labels', 'chinese_labels.txt'), 'r')]

alnum_model = models.create_alnum_model()
alnum_model.load_weights(os.path.join(basedir, 'trained_models', 'alnum_weights.h5'))
alnum_labels = [line.strip() for line in open(os.path.join(basedir, 'labels', 'alnum_labels.txt'), 'r')]

def recognize(chars):
    N = len(chars)
    width, height = chars[0].shape

    chars = numpy.asarray(chars, dtype=numpy.float32).reshape((N, 1, width, height)) / 255.0

    # chinese_classes = np_utils.probas_to_classes(chinese_model.predict(chars[:1], batch_size=1))
    alnum_classes = np_utils.probas_to_classes(alnum_model.predict(chars[1:], batch_size=6))
    
    return ['浙'] + \
           [alnum_labels[cls] for cls in alnum_classes]

