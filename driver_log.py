
"""
Affidavits

 We the undersigned promise that we have in good faith attempted to follow the principles of pair programming.
 Although we were free to discuss ideas with others, the implementation is our own.
 We have shared a common workspace (possibly virtually) and taken turns at the keyboard for the majority of
 the work that we are submitting. Furthermore, any non programming portions of the assignment were done independently.
 We recognize that should this not be the case, we will be subject to penalties as outlined in the course syllabus.
 Yage Jin, Mohit Agarwal, Vivek Kumar
"""

import os
import glob

# Add-on modules
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, \
    TimeDistributed, BatchNormalization
from tensorflow.keras import regularizers

import numpy as np
import matplotlib.pyplot as plt

from lib.fileio import read_data
from lib.kfold_log import split_data
from myclassifier.recurrent_log import handle_k_fold

def main():
    adv_s = 0.01
    len_s = 0.02
    k_fold = 5
    epochs = 15

    ## read data from records folder
    data = read_data(adv_s, len_s)

    ## parse training and testing sets for all folds
    trains, tests = split_data(data, k_fold)

    ## our model
    ## models_rnn[0] - 2 LSTM layers
    ## models_rnn[1] - 1 LSTM layers
    ## models_rnn[2] - 3 LSTM layers
    models_rnn = [
        lambda dim, width, dropout, l2 :
         [(Masking, [], {"mask_value":0.,
                       "input_shape":[None, dim]}),
         (LSTM, [width], {
             "return_sequences":True,
             "kernel_regularizer":regularizers.l2(l2),
             "recurrent_regularizer":regularizers.l2(l2)
             }),
         (BatchNormalization, [], {}),
         (Dropout, [dropout], {}),
         (LSTM, [width], {
             "return_sequences":True,
             "kernel_regularizer":regularizers.l2(l2),
             "recurrent_regularizer":regularizers.l2(l2)
             }),
         (BatchNormalization, [], {}),
         (Dense, [3], {'activation':'softmax',
                             'kernel_regularizer':regularizers.l2(l2)},
            # The Dense layer is not recurrent, we need to wrap it in
            # a layer that that lets the network handle the fact that
            # our tensors have an additional dimension of time.
            (TimeDistributed, [], {}))
         ]
    ]

    handle_k_fold('2 layer 50 nodes 0.01 l2', models_rnn[0], k_fold, 50, 0.2, 0.01, epochs, trains, tests)


if __name__ == '__main__':
    plt.ion()

    main()
