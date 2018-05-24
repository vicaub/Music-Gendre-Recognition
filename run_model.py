from __future__ import print_function, division
import os
import numpy as np
from keras.models import load_model
from mel_spectrogram import extractFeatures

def predictSong(model, songFeatures):
    test = np.asarray(songFeatures)
    print(test.shape)

    test = test.reshape(test.shape[0], 128, 128, 1)

    test = test.astype('float32')
    test /= 255

    return model.predict(test, verbose=1)

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

    songFeatures = extractFeatures(args[0])

    model = load_model('./models/' + options.model)

    output = predictSong(model, songFeatures)

    sumProbabilities = output[0]
    for i in range(1, len(output)):
        sumProbabilities = np.add(sumProbabilities, output[i])

    print(sumProbabilities)

    maxProbabilities = [0 for i in range(10)]
    for i in range(0, len(output)):
        maxProbabilities[np.argmax(output[i])] += 1
    maxProbabilities = np.array(maxProbabilities)
    # maxProbabilities /= sum(maxProbabilities)

    print(maxProbabilities)
    



