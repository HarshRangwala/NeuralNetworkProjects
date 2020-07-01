import glob
import pickle
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from music21 import converter, instrument, note, chord

from plot import plot_history

# suppress warnings due to keras not having fixed deprecation warnings in current version
# see https://github.com/tensorflow/tensorflow/issues/25996 for more details
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_notes():
    """Get all the notes and chords from the midi files in the ./input_music directory."""
    notes = []

#     counter = 0
    for file in glob.glob("input_music/*.mid"):
#         if counter == 10:
#             break
        midi = converter.parse(file)

        print(f"Parsing {file}")

        notes_to_parse = None

        try:            # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:         # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
#         counter += 1

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    """Prepare the sequences used by the Neural Network."""
    sequence_length = 100

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))

    # Normalize input
    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """Create the structure of the neural network."""
    model = Sequential()
    model.add(LSTM(
        256,
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
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def train(model, network_input, network_output):
    """Train the neural network."""
    filepath = "trained_models/weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    history = model.fit(
        network_input,
        network_output,
        epochs=30,
        batch_size=64,
        callbacks=callbacks_list
    )

    return history


if __name__ == '__main__':

    # Pre-processing of data
    notes = get_notes()
    n_vocab = len(set(notes))

    # Data formation
    network_input, network_output = prepare_sequences(notes, n_vocab)

    # Model architecture
    model = create_network(network_input, n_vocab)

    # Training
    history = train(model, network_input, network_output)

    # Plotting
    plot_history(history)
