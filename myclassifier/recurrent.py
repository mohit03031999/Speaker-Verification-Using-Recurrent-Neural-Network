import math
import time
import os.path

import numpy as np

import keras
from keras.callbacks import TensorBoard, History
from keras.utils import np_utils
from sklearn import metrics

import keras.backend as K
import tensorflow as tf

from myclassifier.batchgenerator import PaddedBatchGenerator

import matplotlib.pyplot as plt
import itertools

from lib.buildmodels import build_model

def train_and_evaluate(train, test, model, batch_size=4, epochs=25, name="model"):

    ## create the batches for train annd test
    paddedTrainBatch = PaddedBatchGenerator(train['samples'], train['labels'], batch_size)
    paddedTestBatch = PaddedBatchGenerator(test['samples'], test['labels'], batch_size)

    ## compile the model
    model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    ## train the model
    model.fit(paddedTrainBatch, epochs=epochs, verbose=2)

    ## get predictions and labels
    actual, predicted = get_labels_and_prediction_without_padding(model, paddedTestBatch)

    ## calculate confusion matrix
    m = metrics.confusion_matrix(actual, predicted, labels=np.arange(4))

    ## get frame level accuracy
    frame_result = model.evaluate(paddedTestBatch, verbose=2)

    print('labels:   ', actual)
    print('predicted:', predicted)

    ## calculate file level accuracy
    count = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            count += 1

    return round(1 - frame_result[1], 4), round(1 - count / len(actual), 4), m

def handle_k_fold(name, models_rnn, k_fold, nodes, dropout, l2, epochs, trains, tests):
    print('--------------------------------------------------')
    errs_frame = []
    errs_file = []
    matrixes = None

    ## calculates data for all folds
    for i in range(k_fold):
        ## build new model for each fold
        rnn = build_model(models_rnn(20, nodes, dropout, l2))

        ## get error rate per frame and per file, and the confusion matrix
        err_frame, err_file, matrix = train_and_evaluate(trains[i], tests[i], rnn, epochs=epochs, name=name)
        errs_frame.append(err_frame)
        errs_file.append(err_file)

        ## sum all folds' confusion matrix
        if matrixes is not None:
            matrixes += matrix
        else:
            matrixes = matrix

    print('result for model:', name)
    print('frame avg err:', round(np.mean(errs_frame), 4))
    print('frame std err:', round(np.std(errs_frame), 4))
    print('file avg err: ', round(np.mean(errs_file), 4))
    print('file std err: ', round(np.std(errs_file), 4))
    print(matrixes)
    print('--------------------------------------------------')

## calculate labels and prediction for each file and removed padding
def get_labels_and_prediction_without_padding(model, paddedTestBatch):
    actual_labels = []
    predicted_labels = []
    ## flattening the matrix while categorize the class
    ## output should be 1D array of class indexes
    for i in range(len(paddedTestBatch)):
        ## get examples and labels
        examples, targets = paddedTestBatch[i]

        ## get predictions
        prediction = model.predict(examples)

        ## get the class index while removing paddings
        for j in range(len(targets)):
            actual_label = []
            predicted_label = []
            for k in range(len(targets[j])):
                if np.sum(targets[j][k]) != 0:
                    print(prediction[j][k])
                    actual_label.append(np.argmax(targets[j][k]))
                    predicted_label.append(np.argmax(prediction[j][k]))
            actual_labels.append(np.bincount(np.array(actual_label)).argmax())
            predicted_labels.append(np.bincount(np.array(predicted_label)).argmax())

    return actual_labels, predicted_labels
