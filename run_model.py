from __future__ import print_function, division
import os
import numpy as np
from keras.models import load_model
from mel_spectrogram import extractFeatures
from conv_neural_network import genre_to_label

def predictSong(model, songFeatures):
    test = np.asarray(songFeatures)

    test = test[len(test)//4: len(test) - len(test)//4]

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
        args = [name for name in os.listdir("./testsongs") if os.path.isfile("./testsongs/" + name)]

        print("All test files are selected")
        print(args)
    
    # import the right model
    model = load_model('./models/' + options.model) 

    for filePath in args:
        filePath = "./testsongs/" + filePath
        songFeatures = extractFeatures(filePath)

        output = predictSong(model, songFeatures)

        sumProbabilities = output[0]
        for i in range(1, len(output)):
            sumProbabilities = np.add(sumProbabilities, output[i])
        print("Sum of probabilities")
        label_to_genre = {v: k for k, v in genre_to_label.items()}
        for i in range(len(sumProbabilities)):
            print(label_to_genre[i + 1], sumProbabilities[i])

        maxProbabilities = [0 for i in range(10)]
        for i in range(0, len(output)):
            maxProbabilities[np.argmax(output[i])] += 1
        for i in range(len(maxProbabilities)):
            maxProbabilities[i] /= sum(maxProbabilities)
        print("Max probabilities")
        for i in range(len(maxProbabilities)):
            print(label_to_genre[i + 1], maxProbabilities[i])



