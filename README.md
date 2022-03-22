# Speaker-Verification-Using-Recurrent-Neural-Network
The major objective of this study is to classify a speaker into different categories, so that specific speakers may be granted access to corresponding systems 
and others will not be able to access these systems.

This verification system recognizes and differentiates between speakers by extracting features and characteristics from the speaker's voice and analyzing them.

# Methods
We started the experiment with parsing the audio files into frames with 20 milliseconds frame size and 10 milliseconds advance, which was 160 samples per frame. 
A spectrogram was constructed by adding a hamming window on the frames and applying a Short-Time Fourier Transform (STFT) after, for each of the audio files. 
Once the spectrogram was computed, a mel scale filter with 20 mels was applied on the squared spectrogram in order to reduce the complexity and provide a better
resolution for lower frequency samples.

 We then worked on removing all the noise frames from the mel spectrogram. These noise frames included pauses during speech and silences. A root mean square 
 signal for each audio file was calculated and passed to a Gaussian mixture model, where the RMS signals were categorized into 2 classes, noise and speech. 
 Boolean values were used to label these signals, where true represented speech and false represented noise. Recurrent neural networks were used to train and 
 test on these mel spectrograms using 5-fold cross validation.
