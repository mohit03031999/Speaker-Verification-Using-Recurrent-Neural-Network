'''
@author: us
'''

import librosa
import soundfile
import numpy as np

import lib.rms
from lib.endpointer import Endpointer

## construct spectrogram given recording file name
def construct_spectrogram(filename, adv_s, len_s):
    ## read samples from audio file
    samples, Fs = soundfile.read(filename)
    len_N = int(Fs * len_s)
    adv_N = int(Fs * adv_s)

    ## construct spectrogram
    spectrogram = librosa.stft(samples, hop_length=adv_N, n_fft=len_N, window='hamming', center=False)
    ## square spectrogram
    spectrogram_squared = np.abs(spectrogram)**2

    ## create mel filterbank
    mel_filterbank = librosa.filters.mel(Fs, len_N, n_mels=20, fmin=0, fmax=Fs/2)
    ## construct mel filerbank
    mel_power = np.dot(mel_filterbank, spectrogram_squared)
    ## convert to dB
    mel_spectrogram = 10*np.log10(mel_power + 1e-9)

    ## predict voice for each frame
    rms = lib.rms.get_rms(samples, len_N, adv_N).reshape(-1, 1)
    endpointer = Endpointer(rms)
    labels = endpointer.predict(rms)
    ## remove noise
    mel_spectrogram_voice = np.apply_along_axis(lambda x: np.extract(labels, x), axis=1, arr=mel_spectrogram)

    return mel_spectrogram_voice.transpose()
