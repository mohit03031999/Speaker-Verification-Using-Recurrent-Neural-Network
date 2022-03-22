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

from myclassifier.batchgenerator_log import PaddedBatchGenerator

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
##    frame_result = model.evaluate(paddedTestBatch, verbose=2)

    print('labels:   ', actual)
    print('predicted:', predicted)

    ## calculate file level accuracy
    err = round(1 - (m.trace() / np.sum(m)), 4)

    return 0,err, m

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

        ## sum(log(p)) / T

        ## get the class index while removing paddings
        for j in range(len(targets)):
            predicted_label = []
            for k in range(len(targets[j])):
                if k == 0:
                    if sum(targets[j][k]) == 0:
                        actual_labels.append(3)
                    else:
                        actual_labels.append(np.argmax(targets[j][k]))
                if np.sum(examples[j][k]) != 0:
                    predicted_label.append(prediction[j][k])

            probabilities = np.exp(np.sum(np.log(predicted_label), axis=0) / len(predicted_label))
            max_probability_idx = np.argmax(probabilities)

            print(probabilities)
            if (probabilities[max_probability_idx] >= 0.6):
                predicted_labels.append(max_probability_idx)
            else:
                predicted_labels.append(3)
            

    return actual_labels, predicted_labels

## This method is copied from plot_confusion in confusion.py without the
## section that calculates the confusion matrix
def plot_confusion(cumulative, N, labels):
    # Build the figure and axes
    fig = plt.figure(figsize=(4.5,4.5), dpi=220,
                    facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)


    # Plot the confusion matrix
    # Show the heat map relative to the number of true examples
    # so that the most frequent decisions are always highlighted
    relative = cumulative / cumulative.sum(axis=1)[:,None]
    im = ax.imshow(relative, cmap='Oranges')

    # Label the axes
    tick_marks = np.arange(N)

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(labels, fontsize=3.5, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    # Add counts in boxes
    for i, j in itertools.product(
        range(cumulative.shape[0]), range(cumulative.shape[1])):

        ax.text(j, i, format(cumulative[i, j], 'd') if cumulative[i,j] !=0
                else '.',
                horizontalalignment="center", fontsize=2,
                verticalalignment='center', color= "black")

    fig.set_tight_layout(True)

    return fig, ax, im
