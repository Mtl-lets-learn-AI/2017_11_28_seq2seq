from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np

import helper

# Number of Epochs
epochs = 60
# Batch Size
batch_size = 64  # Batch size for training.
# RNN Size (size of the state vector in encoder and decoder)
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001

source_path = 'data/letters_source.txt'
target_path = 'data/letters_target.txt'

source_sentences = helper.load_data(source_path)
target_sentences = helper.load_data(target_path)


def extract_character_vocab(data):
    # PAD : padding
    # UNK : unkown (often use to talk about something not in the vocabulary) here I don't know
    # GO : begin of decoder(to give something to the RNN as first input)
    # EOS : end of sentence
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']

    set_words = set([character for line in data.split('\n')
                     for character in line])
    int_to_vocab = {word_i: word for word_i,
                    word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int

# Build int2letter and letter2int dicts
# dictionary is different for source and target (not really necessary
# here, but useful in more general situation)
source_int_to_letter, source_letter_to_int = extract_character_vocab(
    source_sentences)
target_int_to_letter, target_letter_to_int = extract_character_vocab(
    target_sentences)


def source_to_seq(text):
    '''Prepare the text for the model'''
    sequence_length = 7
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [source_letter_to_int['<PAD>']] * (sequence_length - len(text))


encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


# Convert characters to ids
source_letter_ids = [[source_letter_to_int.get(letter, source_letter_to_int[
                                               '<UNK>']) for letter in line] for line in source_sentences.split('\n')]
target_letter_ids = [[target_letter_to_int.get(letter, target_letter_to_int[
                                               '<UNK>']) for letter in line] + [target_letter_to_int['<EOS>']] for line in target_sentences.split('\n')]

encoder_inputs = Embedding(encoding_embedding_size, decoding_embedding_size)
encoder = LSTM(rnn_size, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]

decoder_inputs = Embedding(encoding_embedding_size, decoding_embedding_size)
decoder_lstm = LSTM(rnn_size, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(len(target_letter_ids), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')
