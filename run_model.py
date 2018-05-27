from __future__ import print_function, division
import os
import numpy as np
from keras.models import load_model
from mel_spectrogram import extractFeatures
from conv_neural_network import genre_to_label

def predictSong(model, songFeatures):
    test = np.asarray(songFeatures)

    test = test[len(test)//4: len(test) - len(test)//4]

    test = test.reshape(test.shape[0], 128, 128, 1)

    test = test.astype('float32')
    test /= 255

    return model.predict(test, verbose=1)

if __name__ == "__main__":
    
    import optparse
    optparser = optparse.OptionParser()
    optparser.add_option(
        '-m', '--model', type='string',
        help='name of the model to use')

    options, args = optparser.parse_args()

    if not options.model:
        print("No model selected")
        optparser.print_help()
        exit(-1)

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

        label_to_genre = {v: k for k, v in genre_to_label.items()}

        # sumProbabilities = output[0]
        # for i in range(1, len(output)):
        #     sumProbabilities = np.add(sumProbabilities, output[i])
        # print()
        # print("Sum of probabilities")
        # for i in range(len(sumProbabilities)):
        #     print(label_to_genre[i], sumProbabilities[i])

        print()
        print("Sum of probabilities")
        for i in label_to_genre.keys():
            probability = 0
            for square in output:
                probability += square[i]
            print(label_to_genre[i], probability / len(output))
            

        print()

        maxProbabilities = [0 for i in range(len(label_to_genre.keys()))]
        for i in range(0, len(output)):
            maxProbabilities[np.argmax(output[i])] += 1
        for i in range(len(maxProbabilities)):
            maxProbabilities[i] /= len(output)
        print("Max probabilities")
        for i in range(len(maxProbabilities)):
            print(label_to_genre[i], maxProbabilities[i])

        print()


