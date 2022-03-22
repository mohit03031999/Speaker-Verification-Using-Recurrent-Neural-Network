
import numpy as np
from tensorflow.keras.utils import Sequence

class PaddedBatchGenerator(Sequence):
    """PaddedBatchGenerator
    Class for sequence length normalization.
    Each sequence is normalized to the longest sequence in the batch
    or a specified length by zero padding.
    """

    debug = False

    def __init__(self, samples, labels, batch_size=3, shuffle=True):
        """
        PaddedBatchGenerator
        Generate a tensor of examples from the specified corpus.
        :param corpus:   Corpus object, must support get_features, get_labels
        :param utterances: List of utterances in the corpus, e.g. a list
            of files returned by corpus.get_utterances("train")
        :param batch_size: mini batch size
        :param shuffle:  Shuffle instances each epoch
        """
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._epoch = 0
        self.order = np.arange(len(self.samples))

    @property
    def epoch(self):
        return self._epoch

    def __len__(self):
        """len() - Number of batches in data"""
        return int(len(self.samples) / self.batch_size)

    def __getitem__(self, batch_idx):
        """
        Get idx'th batch
        :param batch_idx: batch number
        :return: (examples, targets) - Returns tuple of data and targets for
           specified
        """

        # Hints:  Compute features for each item in batch and keep them in a
        # list.  Then determine longest and fill in missing values with zeros
        # (or other Mask value).

        # handle out of index
        if batch_idx >= len(self):
            raise IndexError

        maxLen = 0
        features = []
        targets = []

        ## get the sorted index given batch index
        indexes = self.order[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]

        ## calculate features and labels for each utterance
        for i in indexes:
            feature = self.samples[i]
            
            ## get all features of the batch
            features.append(feature)

            ## find the longest feature vector in this batch
            if feature.shape[0] > maxLen:
                maxLen = feature.shape[0]

            ## get one hot labels for the batch
            targets.append(self.labels[i])

        paddedBatch = []
        paddedTarget = []

        ## pad both features and labels to max length
        for i in range(self.batch_size):
            input_dim = features[i].shape[1]
            f = list(features[i])

            ## pad feature vectors with all 0s
            while len(f) != maxLen:
                f.append(np.array([0] * input_dim))

            paddedBatch.append(np.array(f))

            ## pad one hot labels with all 0s
            t = list(targets[i])

            if maxLen - len(t) >= 0:
                t += [[0] * 3] * int((maxLen - len(t)))

            paddedTarget.append(np.array(t))

        return np.array(paddedBatch), np.array(paddedTarget)


    def on_epoch_end(self):
        """
        on_epoch_end - Bookkeeping at the end of the epoch
        :return:
        """

        # Change these if you use different variables, otherwise nothing to do.
        # Rather than shuffling the data, I shuffle an index list called order
        # which you would need to create in the constructor.
        self._epoch += 1  # Note next epoch
        if self.shuffle:
            np.random.shuffle(self.order)  # reshuffle the data
