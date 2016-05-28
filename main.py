import sys
import os
import time
import argparse
import matplotlib.pyplot as plt

import cv2
import numpy

from detection import detector
from localization import locator
from segmentation import segmentor
from recognition import recognizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepPR demo')
    parser.add_argument('--display', dest='display',
                        help='Display images',
                        default=False, action='store_true')
    args = parser.parse_args()

    while True:
        filepath = str(input('Please input image file path: ')).strip()

        if filepath.lower() == 'exit' or filepath.lower() == 'quit':
            plt.close('all')
            sys.exit()

        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            print('Not valid file path')
            continue

        origin_image = cv2.imread(filepath)

        if origin_image is None:
            print('Cannot read image')
            continue

        start = time.time()
        license_images = [image for image, score in detector.detect(origin_image)['license']]
        print('Detection time epapsed: %fms' % ((time.time() - start)*1000.0))

        for license_image in license_images:
            seg_start = time.time()
            rects, chars = segmentor.segment(license_image)
            print('Segmentation time epapsed: %fms' % ((time.time() - seg_start)*1000.0))

            if len(chars) != 7:
                print('Illegal Plate License')
                continue

            reg_start = time.time()
            result = recognizer.recognize(chars)
            print('Recognition time epapsed: %fms' % ((time.time() - reg_start)*1000.0))

            print(result)

            if args.display:
                er_image = license_image.copy()
                for rect in rects:
                    cv2.rectangle(er_image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 3)
                chars_image = numpy.hstack(chars)

                plt.close('all')
                figure, axis = plt.subplots(nrows=2, ncols=2)
                plt.suptitle(''.join(result), size=32)
                axis[0][0].imshow(origin_image[:,:,::-1])
                axis[0][1].imshow(license_image[:,:,::-1])
                axis[1][0].imshow(er_image[:,:,::-1])
                axis[1][1].imshow(chars_image, cmap=plt.cm.gray)
                plt.show()

        end = time.time()
        print('total ime elapsed: %fms' % ((end-start)*1000.0))
