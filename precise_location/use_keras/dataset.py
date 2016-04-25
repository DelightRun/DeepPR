#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import cv2
import numpy

basepath = os.path.dirname(os.path.abspath(__file__))


def _load_image(line):
    filename, keypoints = line.split('*')
    filename = filename.strip()
    keypoints = map(int, keypoints.strip().split())

    image = cv2.imread(os.path.join(basepath, 'data', filename+'.jpg'))
    keypoints = numpy.asarray(zip(keypoints[::2], keypoints[1::2]), dtype="float32")

    width, height = image.shape[1], image.shape[0]
    target_width, target_height = 240, 120

    keypoints[:,0] = keypoints[:,0] / width
    keypoints[:,1] = keypoints[:,1] / height
    image = cv2.resize(image, (target_width,target_height))
    image = image.astype("float32")
    image /= 255.0
    image -= 0.5

    return image, keypoints


def load_as_list():
    labelfile = os.path.abspath(os.path.join(basepath, os.pardir, 'data', 'label.txt'))
    return [_load_image(line) for line in open(labelfile, 'r').readlines()]


def load_as_nparray():
    data = load_as_list()
    X, y = zip(*data)
    X = numpy.asarray(X).transpose((0,-1,1,2))
    y = numpy.asarray(y).reshape((len(y), 8))
    return X, y

if __name__ == '__main__':
    load_as_list()
    load_as_nparray()