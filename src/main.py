import configparser

import numpy as np
import pandas as pd

from audiostruct import AudioStruct


def main():
    # Parse config file
    config = configparser.ConfigParser()
    config.read('params.ini')

    # Constants
    DATASET = config['FILE_READ']['DATASET']
    DATA_TYPE = config['FILE_READ']['DATA_TYPE']
    SAVE_SPECS = config['FILE_READ']['SAVE_SPECS']
    audio_dir = DATASET + 'audio/'

    if DATA_TYPE == 'AUDIO_FILES':
        song_representation = AudioStruct(audio_dir)
        spectrograms, song_ids = song_representation.make_spectrograms()

        if SAVE_SPECS:
            np.save('spectrograms.npy', spectrograms)
            np.save('song_ids.npy', song_ids)

    elif DATA_TYPE == 'SPECTROGRAMS':
        spectrograms = np.load(DATASET + 'spectrograms.npy')
        song_ids = np.load(DATASET + 'song_ids.npy')

    else:
        raise ValueError('Argument Invalid: \
                         The options are AUDIO_FILES or SPECTROGRAMS')


