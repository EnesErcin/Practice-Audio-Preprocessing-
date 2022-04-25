"""
    To measure the change in given sound waves there most be a collection of data.
    Device must gather audio recording with an optimal rate, those audios must be saved and analyised autonomosly.

    Visuallisations are for thinking how the underwater recordings can be processed to extract meaningful results.

    The Source of the code : https://github.com/musikalkemist/generating-sound-with-neural-networks/blob/main/12%20Preprocessing%20pipeline/preprocess.py
    -Check Pipline 

    I try to implement the technique Valerio Velardo used in his music genre recognition project.


"""





import numpy as np
import librosa
import os




class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        return signal



class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal.
    """

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram