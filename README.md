# Music Genre Recognition

## Introduction

## Installation

Make sure to have a recent version of python3  with pip3 installed

Then install the python dependencies
```
pip3 install -r requirements.txt
```

Install librosa plugin to read mp3 audio files. (macos only)
```
brew install ffmpeg
```

Download the database
```
wget -O ./gtzan.tar.gz 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
```

Extract the database
```
tar -xzvf gtzan.tar.gz
```

## How to use

First you need to generate the .pickle files corresponding to the feature vectors for the convolutional neural network
```
python3 mel_spectrogram.py
```

Then you need to run the convolutional neural network to create the model
```
python3 conv_neural_network.py [options]

Options:
  -h, --help            show this help message and exit
  -e EPOCS, --epocs=EPOCS
                        number of epocs
  -m MODEL, --model=MODEL
                        name of the model to store
```

To try a model on a new song you need to copy a song in the root folder of the project and change the extention to ".testsong". Then run the following command
```
python3 run_model.py [options]

Options:
  -h, --help            show this help message and exit
  -m MODEL, --model=MODEL
                        name of the model to use
```

It will store the model at the end of the process in the folder ./models/

## TODO

clean conv_neural_network.py = separate functions, provide more running options, and be able to retrieve stored model
Remove beginning and end of testing songs