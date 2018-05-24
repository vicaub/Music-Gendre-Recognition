import os
import librosa
import numpy as np
import pickle
# import matplotlib.pyplot as plt
# import librosa.display

WINDOW_DURATION = 0.02 #secconds
label_list = []

def storeArray(featuresArray, ppFilePath):
    print('storing pp file: ' + ppFilePath)

    f = open(ppFilePath, 'wb')
    pickle.dump(featuresArray, f)
    f.write(pickle.dumps(featuresArray))
    f.close()

def extractFeatures(audioPath):
    print('Prepossessing ' + audioPath)

    Y, sr = librosa.load(audioPath)
    SOUND_SAMPLE_LENGTH = len(Y)
    WINDOW_SAMPLE_LENGTH = int(WINDOW_DURATION * sr)

    S = librosa.feature.melspectrogram(Y, sr=sr, hop_length=WINDOW_SAMPLE_LENGTH, n_mels=128)
    shape = S.shape

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

if __name__ == "__main__":
    label_list = [name for name in os.listdir("./genres") if os.path.isdir("./genres/" + name)]

    for label in label_list:
        path = "./genres/" + label
        for song in os.listdir(path):
            if os.path.isfile(path + "/" + song) and not song.endswith(".pickle"):
                audioPath = path + "/" + song
                ppFilePath = path + "/" + song + ".pickle"
                if ppFilePath not in os.listdir(path):
                    squares = extractFeatures(audioPath)
                    storeArray(squares, ppFilePath)
                    

