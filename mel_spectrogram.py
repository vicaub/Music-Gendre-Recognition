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

    # g = open(ppFilePath, 'rb')
    # recup = pickle.load(g)

    # print(recup)

def prepossessingAudio(audioPath, ppFilePath):
    print('Prepossessing ' + audioPath)

    featuresArray = []
    Y, sr = librosa.load(audioPath)
    SOUND_SAMPLE_LENGTH = len(Y)
    WINDOW_SAMPLE_LENGTH = int(WINDOW_DURATION * sr)
    # print("sr", sr, "WINDOW_SAMPLE_LENGTH", WINDOW_SAMPLE_LENGTH)

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

    # print(len(squares))
    # print(len(squares[0]))
    # print(len(squares[0][0]))

    return squares

if __name__ == "__main__":
    label_list = [name for name in os.listdir("./genres") if os.path.isdir("./genres/" + name)]
    print("label_list", label_list)

    for label in label_list:
        path = "./genres/" + label
        for song in os.listdir(path):
            if os.path.isfile(path + "/" + song) and not song.str.endswith(".pickle"):
                audioPath = path + "/" + song
                ppFilePath = path + "/" + song + ".pickle"
                if ppFilePath not in os.listdir(path):
                    squares = prepossessingAudio(audioPath, ppFilePath)
                    storeArray(squares, ppFilePath)
                    

    # file_path = "blues.00000.au"
    # ppFilePath = "test_mfcc"
    # prepossessingAudio(file_path, ppFilePath)

