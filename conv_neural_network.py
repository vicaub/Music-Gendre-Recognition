import os
import pickle
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

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

if __name__ == "__main__":

    # Parameters
    # learning_rate = 0.001
    # training_iters = 100000
    # batch_size = 64
    # display_step = 1
    train_size = 0.7

    # Network Parameters
    # n_input = 599 * 128
    # n_input = 599 * 128*2
    # n_classes = 10
    # dropout = 0.75  # Dropout, probability to keep units

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
                        print("retrieved song:", song)

                        

    data = np.asarray(data)
    labels = np.asarray(labels)
    print("labels shape", labels.shape)
    print("data shape", data.shape)
    dataLength = len(data)
    # data = data.reshape((data.shape[0], n_input))

    # #Hack
    # data = np.random.random((1000, n_input))
    # labels = np.random.random((1000, 10))

    # Shuffle data
    permutation = np.random.permutation(dataLength)
    data = data[permutation]
    labels = labels[permutation]

    # Split Train/Test
    x_train = data[:int(train_size * dataLength)]
    y_train = labels[:int(train_size * dataLength)]

    x_test = data[int(train_size * dataLength):]
    y_test = labels[int(train_size * dataLength):]

    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 128, 128


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

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
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save("./models/cnn.h5")
