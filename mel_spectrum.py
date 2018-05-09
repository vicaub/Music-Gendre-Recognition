import os
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
import librosa.display

WINDOW_DURATION = 0.1


def prepossessingAudio(audioPath, ppFilePath):
    print('Prepossessing ' + audioPath)

    featuresArray = []
    Y, sr = librosa.load(audioPath)
    SOUND_SAMPLE_LENGTH = len(Y)
    WINDOW_SAMPLE_LENGTH = int(WINDOW_DURATION * sr)

    S = librosa.feature.melspectrogram(Y, sr=sr, hop_length=WINDOW_SAMPLE_LENGTH, n_mels=1000)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

    # for i in range(int(SOUND_SAMPLE_LENGTH // WINDOW_SAMPLE_LENGTH)):
    #     y = Y[i*WINDOW_SAMPLE_LENGTH: (i+1)*WINDOW_SAMPLE_LENGTH]

    #     # Let's make and display a mel-scaled power (energy-squared) spectrogram
    #     S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    #     # Convert to log scale (dB). We'll use the peak power as reference.
    #     log_S = librosa.amplitude_to_db(S)
    #     # Finally compte the MFCC
    #     mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=1)[0]

    #     featuresArray.append(mfcc)

    #     print(mfcc)

    #     # featuresArray.append(S)



    # print('storing pp file: ' + ppFilePath)
    

    # f = open(ppFilePath, 'wb')
    # pickle.dump(featuresArray, f)
    # # f.write(pickle.dumps(featuresArray))
    # f.close()

    # g = open(ppFilePath, 'rb')
    # recup = pickle.load(g)

    # print(recup)

if __name__ == "__main__":
    file_path = "blues.00000.au"
    ppFileName = "test_mfcc"
    prepossessingAudio(file_path, ppFileName)

