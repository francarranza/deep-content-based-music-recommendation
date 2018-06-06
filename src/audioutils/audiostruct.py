import os
import librosa
import numpy as np


# @Class: MelSpectrogram
# @Description:
#  Class to read .mp3 files and export the songs as MelSpectrograms
class AudioStruct(object):
    def __init__(self, audio_dir):
        # Constants
        self.audio_dir = audio_dir
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = 128
        self.window_size = 3
        self.offset = 3

    def make_spectrograms(self):
        song_ids = [s.split('.')[0] for s in os.listdir(self.audio_path)]
        song_ids.sort()

        # Create spectrograms dir
        try:
            os.mkdir(self.spectrograms_dir)
        except OSError:
            pass

        spectrograms = []
        for i, song in enumerate(song_ids):
            filename = self.audio_path + song + '.mp3'
            signal, sampling_rate = librosa.load(
                filename, duration=self.duration, offset=self.offset)

            # Make spectrogram
            S = librosa.feature.melspectrogram(
                signal, sr=sampling_rate, n_mels=self.n_mels, n_fft=self.n_fft)
            S_log = librosa.logamplitude(S, ref_power=np.max)
            spectrograms.append(S_log)

            print("Songs remaining:", len(song_ids) - i)

        return spectrograms, song_ids

