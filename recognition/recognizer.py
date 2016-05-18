import os
from keras.utils import np_utils

import models

chinese_model = models.create_chinese_model()
chinses_model.load_weights(os.path.join('trained_models', 'chinese_weights.h5'))
chinese_labels = [line.strip() for line in open(os.path.join('labels', 'chinese_labels.txt'), 'r')]

alnum_model = models.create_alnum_model()
alnum_model.load_weights(os.path.join('trained_models', 'alnum_weights.h5'))
alnum_labels = [line.strip() for line in open(os.path.join('labels', 'alnum_labels.txt'), 'r')]

def recognize(char_imgs):
    chinese_imgs = numpy.asarray(char_imgs[:1]).astype('float32') / 255.0
    classes = np_utils.to_categorical(chinese_model.predict(chinese_imgs))
    chinese = [chinese_labels[cls] for cls in classes]

    alnum_imgs = numpy.asarray(char_imgs[1:]).astype('float32') / 255.0
    classes = np_utils.probas_to_classes(alnum_model.predict(alnum_imgs, batch_size=6))
    alnums = [alnum_labels[cls] for cls in classes]

    return chinese + alnums
