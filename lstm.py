""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization, Embedding
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils, plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import os

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

        print(len(notes))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 25  # 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    #network_input = numpy.reshape(network_input, (n_patterns, sequence_length))
    # normalize input
    network_input = network_input / float(n_vocab)

    print(network_input)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    EMB_DIM = 300
    model = Sequential()
    #model.add(Embedding(input_dim=n_vocab + 1, output_dim=EMB_DIM, input_length=25))
    model.add(LSTM(
        512,
        #input_shape=(network_input.shape[1], EMB_DIM),
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(BatchNormalization())
    model.add(LSTM(512, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))

    """model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))"""

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    plot_model(model, to_file='full_model.png', show_shapes=True)
    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    tb = TensorBoard(log_dir='./logs')
    callbacks_list = [checkpoint, tb]

    #model.load_weights('new-weights.hdf5')

    history = model.fit(network_input, network_output, epochs=200, batch_size=512, callbacks=callbacks_list)


    losses = history.history.get('loss')

    numpy.save('losses1', losses)
    model.save_weights('new-weights.hdf5')


if __name__ == '__main__':
    train_network()

    loss = numpy.load('losses1.npy')
    import matplotlib.pyplot as plt

    plt.plot(loss)

    plt.show()


