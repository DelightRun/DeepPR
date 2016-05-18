import os
import numpy
import pickle
import cv2

with open(os.path.join('data', 'trainx.dat'), 'rb') as f:
    f.read(16)
    X = numpy.array(list(f.read()))

X = X.reshape(16424, 1, 20, 20).astype('float32')
X /= 255.0
X -= numpy.mean(X, axis = 0)
X /= numpy.std(X, axis = 0)

chars = ['' for _ in range(65)]
with open(os.path.join('data', 'trainy.dat'), 'rb') as f:
    f.read(12)
    for i in range(len(chars)):
        chars[i] = chr(int.from_bytes(f.read(2), byteorder='little'))
    labels = numpy.array(list(f.read()))

with open(os.path.join('labels', 'alnum_labels.txt'), 'w') as f:
    f.write('\n'.join(chars[:34]))
with open(os.path.join('labels', 'chinese_labels.txt'), 'w') as f:
    f.write('\n'.join(chars[34:]))

alnum_indices = numpy.array([i for i in range(labels.shape[0]) if labels[i] <= 33])
chinese_indices = numpy.array([i for i in range(labels.shape[0]) if not labels[i] <= 33])

alnum_X = numpy.asarray([[cv2.resize(X[i][0], (50,50))] for i in alnum_indices])
alnum_labels = numpy.asarray([labels[i] for i in alnum_indices])

print('number of alnum characters: ', alnum_labels.shape[0])
with open(os.path.join('data', 'alnum_data.pkl'), 'wb') as f:
    pickle.dump((alnum_X, alnum_labels), f)

chinese_X = numpy.asarray([[cv2.resize(X[i][0], (50,50))] for i in chinese_indices])
chinese_labels = numpy.asarray([labels[i]-34 for i in chinese_indices])

print('number of chinese characters: ', chinese_labels.shape[0])
with open(os.path.join('data', 'chinese_data.pkl'), 'wb') as f:
    pickle.dump((chinese_X, chinese_labels), f)
