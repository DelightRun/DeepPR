#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import cv2
import numpy

basepath = os.path.dirname(__file__)

def _load_image(line):
    filename, keypoints = line.split('*')
    filename = filename.strip()
    keypoints = map(int, keypoints.strip().split())

    image = cv2.imread(os.path.join(basepath, 'data', filename+'.jpg'))
    keypoints = numpy.asarray(zip(keypoints[::2], keypoints[1::2]), dtype="float32")

    width, height = image.shape[1], image.shape[0]
    targetWidth, targetHeight = 240, 120

    keypoints[:,0] = keypoints[:,0] / width
    keypoints[:,1] = keypoints[:,1] / height
    image = cv2.resize(image, (targetWidth,targetHeight))
    image = image.astype("float32")
    image /= 255.0
    image = image - image.mean(axis=(0,1))

    return (image, keypoints)

def load_as_list():
    return [_load_image(line) for line in open(os.path.join(basepath,'label.txt'),'r').readlines()]

def load_as_nparray():
    data = load_as_list()
    X, y = zip(*data)
    X = numpy.asarray(X).transpose((0,-1,1,2))
    y = numpy.asarray(y).reshape((len(y), 8))
    return X, y
