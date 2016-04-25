#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from precise_location.keras.models import Sequential, Graph
from keras.layers import Layer, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam


class Identity(Layer):
    def get_output(self, train=False):
        return self.get_input(train)


def build_residual_block(name, input_shape, conv_shape, n_skip=2):
    block = Graph()
    input_layer = 'x'
    block.add_input(input_layer, input_shape=input_shape)
    block.add_node(Identity(), name=name+'-Identity', input=input_layer)

    prev_layer = input_layer
    for i in range(n_skip):
        layer_name = name + '-Layer%d' % i

        block.add_node(Convolution2D(*conv_shape), name=layer_name+'-Conv', input=prev_layer+'-ReLU')
        block.add_node(BatchNormalization(axis=1), name=layer_name+'-BatchNorm', input=layer_name+'-Conv')
        block.add_node(Activation('relu'), name=layer_name+'ReLU', input=layer_name+'BatchNorm')

        prev_layer = layer_name

    block.add_output(name=name+'-Output', inputs=[name+'-Identity', prev_layer], merge_mode='sum')

    return block


def ResidualNetwork():
    pass


def VGG():
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3,120,240)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    for _ in range(2):
        model.add(Convolution2D(128, 3, 3, border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for _ in range(2):
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    for _ in range(2):
        model.add(Convolution2D(512, 3, 3, border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8))

    model.compile(loss='mse', optimizer=Adam())
    return model


def ConvolutionalNetwork():
    model = Sequential()

    model.add(Convolution2D(4, 7, 7, border_mode='same', input_shape=(3, 120, 240)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Convolution2D(4, 7, 7, border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    for _ in range(2):
        model.add(Convolution2D(8, 5, 5, border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    for _ in range(2):
        model.add(Convolution2D(16, 3, 3, border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(8))

    model.compile(loss='mse', optimizer=Adam())
    return model
