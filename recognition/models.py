from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

nb_filters = 32
nb_pool = 2
nb_conv = 3

def create_chinese_model():
    model = Sequential()

    model.add(Convolution2D(nb_filters,nb_conv,nb_conv,
                        border_mode='valid',
                        input_shape=(1,50,50)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(31))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                                optimizer='adadelta',
                                metrics=['accuracy'])

    return model

def create_alnum_model():
    model = Sequential()

    model.add(Convolution2D(nb_filters,nb_conv,nb_conv,
                        border_mode='valid',
                        input_shape=(1,50,50)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(34))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                                optimizer='adadelta',
                                metrics=['accuracy'])
    return model
