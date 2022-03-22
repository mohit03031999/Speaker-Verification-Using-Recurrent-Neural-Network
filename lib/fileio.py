'''
@author: us
'''

import os
import glob

from lib.spectrogram import construct_spectrogram

## read all files given folder
def parse_data(folder, i, data, adv_s, len_s):
    total = 0

    for filename in glob.glob("records/" + folder + "/*.wav"):
        ## construct spectrogram for each recording
        spectrogram = construct_spectrogram(filename,  adv_s, len_s)

        total += spectrogram.shape[0]

        ## add spectrogram to data
        if i in data.keys():
            data[i].append(spectrogram)
        else:
            data[i] = [spectrogram]

    return total

## read data from records folder
def read_data(adv_s, len_s):
    ## folders to read from
    folders = ['jacob', 'mohit', 'vivek', 'other']
    data = {}
    total = 0
    for i in range(len(folders)):
        ## read each folder when parsing other folder
        if i == 3:
            for folder in os.listdir('records/' + folders[i] + '/'):
                total += parse_data(folders[i] + '/' + folder, i, data, adv_s, len_s)
        else:
            total += parse_data(folders[i], i, data, adv_s, len_s)
    return data
