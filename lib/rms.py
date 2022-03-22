'''
@author: mroch
'''

import librosa.util
import numpy as np

def get_rms(samples, len_N, adv_N, rms_floor=-50):
    """
    get_rms(samples, rms_floor)
    Given sample data,
    Read Nsamples of data (from all channnels) from file filename
    starting at sample start_sample.  If Nsamples < 0, data to the
    end of the file are returned.   Values < rms_floor are set to
    rms_floor to prevent -Inf for very small intensities

    If start_sample is None, reads from current position

    Returns RMS signal of data based on current framing parameters
    """


    # frame the signal
    frames = librosa.util.frame(samples, len_N, adv_N)

    # Convert to RMS, add small offset to avoid log 0
    eps = 1e-7
    rms = 10 * np.log10(np.mean(frames ** 2, axis=0) + eps)

    # Set a floor on the energy
    rms[rms < rms_floor] = rms_floor

    return rms
