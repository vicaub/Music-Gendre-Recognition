import os
import pickle
import numpy as np


genre_to_label = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}

def storeModel(model, name):
    directory = "./models"
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save("./models/"+ name + ".h5")

def prepareData():
    pass

def musicModel():
    pass
def imageModel():
    pass

def makeModel(numberEpocs, modelName):
    import keras
    from keras import backend as K
    from keras.models import Sequential, Model
    from keras.layers import Dense, TimeDistributed, LSTM, Dropout, Activation, Flatten, Convolution2D, Conv2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import ELU

    train_size = 0.7
    validation_size = 0.2
    test_size = 0.1

    # Load data
    data = []
    labels = []
    label_list = [name for name in os.listdir("./genres") if os.path.isdir("./genres/" + name)]
    for label in label_list:
        path = "./genres/" + label
        for song in os.listdir(path):
            if os.path.isfile(path + "/" + song) and song.endswith(".pickle"):
                    with open(path + "/" + song, 'rb') as f:
                        content = f.read()
                        squares = pickle.loads(content)
                        data += squares
                        labels += [genre_to_label[label] for i in range(len(squares))]
    print("imported", len(labels), "song features")

                        

    data = np.asarray(data)
    labels = np.asarray(labels)
    print("labels shape", labels.shape)
    print("data shape", data.shape)
    dataLength = len(data)

    # Shuffle data
    permutation = np.random.permutation(dataLength)
    data = data[permutation]
    labels = labels[permutation]

    # Split Train/Test
    x_train = data[:int(train_size * dataLength)]
    y_train = labels[:int(train_size * dataLength)]

    x_validation = data[int(train_size * dataLength): int((train_size + validation_size) * dataLength)]
    y_validation = labels[int(train_size * dataLength): int((train_size + validation_size) * dataLength)]

    x_test = data[int((train_size + validation_size) * dataLength):]
    y_test = labels[int((train_size + validation_size) * dataLength):]


    batch_size = 128
    num_classes = 10
    epochs = numberEpocs

    # input image dimensions
    img_rows, img_cols = 128, 128


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_validation = x_validation.reshape(x_validation.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_validation = x_validation.reshape(x_validation.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_validation = x_validation.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_validation /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_validation.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_validation = keras.utils.to_categorical(y_validation, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    
    if modelName == "music":
    
        nb_filters = 32  # number of convolutional filters to use
        pool_size = (2, 2)  # size of pooling area for max pooling
        kernel_size = (3, 3)  # convolution kernel size
        nb_layers = 4

        model = Sequential()
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        for layer in range(nb_layers-1):
            model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
            model.add(BatchNormalization())
            model.add(ELU(alpha=1.0))  
            model.add(MaxPooling2D(pool_size=pool_size))
            model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        model.compile(loss='categorical_crossentropy',
                optimizer='adam',
        metrics=['accuracy'])


    #############################

    if modelName == "image":

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])


    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_validation, y_validation))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return model

if __name__ == "__main__":

    import optparse
    optparser = optparse.OptionParser()
    optparser.add_option(
        '-e', '--epocs', type='int', default=12,
        help='number of epocs')
    optparser.add_option(
        '-m', '--model', type='string',
        help='name of the model to store')

    options, args = optparser.parse_args()

    if not options.model:
        print("No model provided")
        optparser.print_help()
        exit(-1)

    model = makeModel(options.epocs, options.model)

    storeModel(model, options.model)
