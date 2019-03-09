# Contemporary Topics in AI
## Computational Creativity - Music Generation

Below you can read the README of the original github

### Installation
Virtual environments work a little bit different on windows. This guide is written for linux OS.

* make a python virtual environment with `python3 -m venv ./venv`
* activate your virtual environment with `source venv/bin/activate`
* install the needed packages with `pip install -r requirements.txt`

### Train
To train a model, change one of the provided database folders (or download your own) into `midi_songs`.
Then simply run `python3 lstm.py`

### Predict
The training will save all better versions of the model. To predict a new song, change one of the weights (watch out for overfitting) to `new-weights.hdf5`.
Then simply run `python3 predict.py`
The generated result will be a midi file `test-output.mid`. You can play it with for example windows media player or other programs that can read midi.

### Other files
* In `data` the program saves a notes dictionary, you can ignore this.
* In `demo` you can find the demo used in the presentation.
* In `documents` you can find pdf files of the presentation, report and papers that we read.
* In `logs` the log files for TensorBoard will be saved.
* In `losses` the loss files of previous training sessions are saved.
* In `midi_songs_...` you can find the different datasets.
* In `pretrained_weights` you can find some of the results of previous training sessions.
* In `results` you can find some midi, some wav and some Sibelius files. The sibelius files are created to inspect the generated score and not only listen to the music, however the Sibelius program will need to be installed to be able to open these files.
* In `visualizations` you can find the graphs used in the presentation.

# Classical Piano Composer

This project allows you to train a neural network to generate midi music files that make use of a single instrument

## Requirements

* Python 3.x
* Installing the following packages using pip:
	* Music21
	* Keras
	* Tensorflow
	* h5py

## Training

To train the network you run **lstm.py**.

E.g.

```
python lstm.py
```

The network will use every midi file in ./midi_songs to train the network. The midi files should only contain a single instrument to get the most out of the training.

**NOTE**: You can stop the process at any point in time and the weights from the latest completed epoch will be available for text generation purposes.

## Generating music

Once you have trained the network you can generate text using **predict.py**

E.g.

```
python predict.py
```

You can run the prediction file right away using the **weights.hdf5** file
