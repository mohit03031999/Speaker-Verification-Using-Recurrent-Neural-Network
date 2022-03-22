'''
@author: mroch
'''

from sklearn.mixture import GaussianMixture
import numpy as np


class Endpointer:
    """
    Endpointer
    Unsupervised voice activity detector

    Uses a GMM to learn a two class distribution of RMS energy vectors

    """
    Nclasses = 2    # two classes, speech/noise

    def __init__(self, train):
        """
        Endpointer(train) - Train a two source separation model for
        detecting speech based on an N example by D dimensional training matrix
        (e.g., RMS energy Nx1)

        :param train:  Numpy N x D matrix of N examples of D dimensional features
        """

        self.gmm = GaussianMixture(self.Nclasses)
        self.gmm.fit(train)

        # Determine which mixture was which
        self.mixture_labels = dict()

        dim = train.shape[1]   # dimensionality of feature

        # Identify the mixture with more energy.  For scalars this is easy.
        # For vectors we can take some estimate of the energy (e.g. 0 Hz dB measurement
        # in a spectrum) or we could take the largest magnitude vector which we will do here.
        # As we are concerned about the highest intensity, we don't want very quiet
        # measurements with large negative numbers to contribute to this, so we will
        # base on measurement of vector norm on modified means that have values >= 0
        smallest = np.min(self.gmm.means_)
        adjmeans = self.gmm.means_ - smallest
        magnitudes = np.linalg.norm(adjmeans, axis=1)

        for extremumfn, label in [ [np.argmin, "noise"], [np.argmax, "speech"]]:
            # Pick the smallest/largest magnitude mixture and assign it to
            # the proper category.
            self.mixture_labels[label] = extremumfn(magnitudes)

    def predict(self, features):
        """
        predict
        :param features: Numpy N x D matrix of N examples of D dimensional features
        :return: binary vector, True for frames classified as speech
        """
        """predict - Extract RMS from file using the same
        framing parameters as the constructor and return vector
        of booleans where True indicates that speech occurred
        """

        # Get mixture membership predictions
        decisions = self.gmm.predict(features)

        # Convert to speech predicate vector
        speech = decisions == self.mixture_labels["speech"]

        return speech
