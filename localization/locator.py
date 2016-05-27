#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Changxu Wang <wang_changxu@icloud.com>
#
# Distributed under terms of the MIT license.

import PyTorchHelpers

import os
import sys
import cv2
import numpy

basedir = os.path.dirname(os.path.realpath(__file__))
model_name = 'model-34.t7'

Model = PyTorchHelpers.load_lua_class('model.lua', 'Model')
model = Model(model_name)

def locate(images):
    resized_images = [cv2.resize(image, (448, 224)) for image in images]
    input_images = numpy.array(resized_images, dtype=numpy.float32) / 255.0
    output = model.forward(input_images).asNumpyTensor()

    output_images = []
    for i, image in enumerate(images):
        width, height = image.shape[:2]

        keypoints = output[i].reshape((4,2))
        keypoints[:, 0] *= width
        keypoints[:, 1] *= height
        keypoints = numpy.round(keypoints)

        vertex = numpy.array([[width-1, 0], [width-1, height-1], [0, height-1], [0,0]], dtype=numpy.float32)

        M = cv2.getPerspectiveTransform(keypoints, vertex)
        output_images.append(cv2.warpPerspective(image, M, (width, height)))

    return output_images

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    if len(sys.argv) != 2:
        print('Usage: python locator.py image')
        exit()

    origin = cv2.imread(sys.argv[1])
    output = locate([origin, ])[0]

    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.imshow(origin[:,:,::-1])
    ax2.imshow(output[:,:,::-1])
    plt.show()
