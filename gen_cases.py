import numpy as np
import os
from six.moves import cPickle

save_dir = '../data/'
save_model_dir = '../data/models/'
seq_length = 30


def sample(preds, temperature=1.0):


    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


if __name__ == "__main__":
    words_number = 30  # number of words to generate
    seed_sentences = " is murder or fraud legal"  # seed sentence to start the generating.

    # initiate sentences
    generated = ''
    sentence = []

    print("loading vocabulary...")
    vocab_file = os.path.join(save_dir, "words_vocab.pkl")

    with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

    vocab_size = len(words)

    from keras.models import load_model

    # load the model
    print("loading model...")
    model = load_model(save_model_dir + "/" + 'my_model_generate_sentences.h5')


    # we shate the seed accordingly to the neural netwrok needs:
    for i in range(seq_length):
        sentence.append("a")

    seed = seed_sentences.split()

    for i in range(len(seed)):
        sentence[seq_length - i - 1] = seed[len(seed) - i - 1]

    generated += ' '.join(sentence)

    # the, we generate the text
    for i in range(words_number):
        # create the vector
        x = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(sentence):
            x[0, t, vocab[word]] = 1.

        # calculate next word
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.33)
        next_word = vocabulary_inv[next_index]

        # add the next word to the text
        generated += " " + next_word
        # shift the sentence by one, and and the next word at its end
        sentence = sentence[1:] + [next_word]

    # print the whole text
    print(generated)