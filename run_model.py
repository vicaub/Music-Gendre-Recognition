from __future__ import print_function, division
import os
import numpy as np
import librosa
from scipy.io import wavfile
from scipy.signal import correlate, hamming
from sklearn.externals import joblib
from keras.models import load_model


WINDOW_DURATION = 0.02 #secconds


def prepossessingAudio(audioPath):
    print('Prepossessing ' + audioPath)

    Y, sr = librosa.load(audioPath)
    SOUND_SAMPLE_LENGTH = len(Y)
    WINDOW_SAMPLE_LENGTH = int(WINDOW_DURATION * sr)

    S = librosa.feature.melspectrogram(Y, sr=sr, hop_length=WINDOW_SAMPLE_LENGTH, n_mels=128)
    shape = S.shape
    # print("shape", shape)

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=sr, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram')
    # plt.tight_layout()
    # plt.show()


    S = np.transpose(S)
    squares = []
    i = 0
    while i < shape[1] - shape[0]:
        squares.append(S[i: i + shape[0]])
        i += shape[0]

    return squares


def predictSong(model, songFeatures):
    return model.predict(songFeatures[0], verbose=1)

if __name__ == "__main__":
    
    import optparse
    optparser = optparse.OptionParser()
    optparser.add_option(
        '-m', '--model', type='string', default="cnn.h5",
        help='name of the model to use')

    options, args = optparser.parse_args()

    if len(args) == 0:
        print("No FILENAME provided")
        optparser.print_help()
        exit(-1)

    songFeatures = prepossessingAudio(args[0])

    model = load_model('./models/' + options.model)

    predictSong(model, songFeatures)