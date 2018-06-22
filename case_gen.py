from __future__ import print_function
#import Keras library
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy

#import spacy, and spacy french model
# spacy is used to work on text
import spacy
nlp = spacy.load('en')
#import other libraries
import numpy as np
import os
import collections
from six.moves import cPickle
#define parameters used in the tutorial
from tqdm import tqdm

data_dir = '../data/all_cases.txt'# data directory containing raw texts
save_dir = '../data/'
save_model_dir = '../data/models/'

file_list = []
vocab_file = os.path.join(save_dir, "words_vocab.pkl")
sequences_step = 1 #step to create sequences
given_length = 30 # length of given words


def create_word_list(path):

    print('Starting wordlist')
    word_list = []
    count = 0

    with open(path) as f:
        for line in tqdm(f):
            if count > 100:
                break
            if count % 100 == 0:
                print("Read {} sentences".format(count))
            words = map(lambda x:x.lower(), line.strip().split())
            word_list = word_list + words
            count = count +1

    print ('Done with wordlist')
    return word_list


def create_vocab(out_path, word_list):

    print('Starting create vocab')
    # count the number of words
    word_counts = collections.Counter(word_list)

    # Mapping from index to word : that's the vocabulary
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Mapping from word to index
    voca = {x: i for i, x in enumerate(vocabulary_inv)}
    words_dict = [x[0] for x in word_counts.most_common()]

    # size of the vocabulary
    voca_size = len(words_dict)
    print("vocab size: ", voca_size)

    # save the words and vocabulary
    with open(os.path.join(vocab_file), 'wb') as f:
        cPickle.dump((words_dict, voca, vocabulary_inv), f)

    print ('Done with create vocab')
    return voca_size, voca, words_dict


def create_given_and_next_word_training_data(word_list):

    print('Starting sequence and Y words')

    given_words_list = []
    next_word_for_given = []

    for i in range(0, len(word_list) - given_length, sequences_step):
        given_words_list.append(word_list[i: i + given_length])
        next_word_for_given.append(word_list[i + given_length])

    print ('Done with sequence and Y words')
    return given_words_list, next_word_for_given


def create_X_and_Y_for_train(vocab, vocab_size, word_list):

    print('Starting matrix creation')

    sequences, next_words = create_given_and_next_word_training_data(word_list)

    print('len of sequence {} '.format(len(sequences)))
    X = np.zeros((len(sequences), given_length, vocab_size), dtype=np.bool)
    y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, word in enumerate(sentence):
            X[i, t, vocab[word]] = 1
        y[i, vocab[next_words[i]]] = 1

    print ('Done with matrix creation')
    return X, y


def bidirectional_lstm_model(seq_length, vocab_size):

    print('Starting model creation')

    lstm_cells = 256  # no. of LSTM cells
    seq_length = given_length  # sequence length
    learning_rate = 0.001  # learning rate

    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(lstm_cells, activation="relu"), input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate)
    callbacks = [EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    print("model built!")
    print ('Done with model creation')

    return model


if __name__ == "__main__":
    '''
    Create wordlist
    Create vocab
    Create dictionaries word - index index-word
    Create X and Y
    Create model
    Train model
    '''
    word_list = create_word_list(data_dir)
    vocab_size, vocab, words = create_vocab(save_dir, word_list)
    X, y = create_X_and_Y_for_train(vocab, vocab_size, word_list)
    model = bidirectional_lstm_model(given_length, vocab_size)

    print(model.summary())

    batch_size = 512  # minibatch size
    num_epochs = 10  # number of epochs

    callbacks = [EarlyStopping(patience=4, monitor='val_loss'),
                 ModelCheckpoint(filepath=save_model_dir + "/" + 'my_model_gen_sentences.{epoch:02d}-{val_loss:.2f}.hdf5', \
                                 monitor='val_loss', verbose=0, mode='auto', period=2)]
    # fit the model
    history = model.fit(X, y,
                     batch_size=batch_size,
                     shuffle=True,
                     epochs=num_epochs,
                     callbacks=callbacks,
                     validation_split=0.1)

    # save the model
    model.save(save_dir + "/" + 'my_model_generate_sentences.h5')


















