# Music-Gendre-Recognition

## Introduction

## Installation

Make sure to have a recent version of python3  with pip3 installed

Then install the python dependencies

```
pip3 install -r requirements.txt
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
python3 conv_neural_network.py
```

It will store the model at the end of the process in the folder ./models/