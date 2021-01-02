import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import nltk
import pickle
from keras.models import load_model
from tensorflow.keras.layers import RNN
from src.LSTM_BiLSTM.attention import AttentionLayer


def text_cleaner(text, contraction_mapping, stop_words):

    # Convert everything to lowercase.
    newString = text.lower()

    # Remove HTML tags.
    newString = BeautifulSoup(newString, "lxml").text

    # Remove trivial symbols.
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"', '', newString)

    # Expand the contraction of words.
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])

    # Remove 's, punctuation, and special characters.
    newString = re.sub(r"'s\b", "", newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)

    # Tokenize the string and remove stop words.
    tokens = [w for w in newString.split() if not w in stop_words]

    # Remove the words too short.
    long_words = []
    for i in tokens:
        if len(i) >= 3:
            long_words.append(i)

    return (" ".join(long_words)).strip()


def decode_sequence(input_seq, encoder_model, decoder_model, target_word_index, reverse_target_word_index):

    max_len_text = 500
    max_len_summary = 50
    latent_dim = 500
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Chose the 'start' word as the first word of the target sequence.
    target_seq[0, 0] = target_word_index['start']

    # Keep decoding until the stop condition is true.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        # Get the next decoded token.
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        # If the token isn't the special token "end", append it into the decoded sentence.
        if(sampled_token != 'end'):
            decoded_sentence += ' ' + sampled_token

        # If the token is the special token "end" or the length has reached maximum, break the loop.
        if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary - 1)):
            stop_condition = True

        # Update the target sequence of length 1 for the prediction of next token.
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states.
        e_h, e_c = h, c

    return decoded_sentence


def summarize_BiLSTM(text):
    nltk.data.path.append('./nltk_data')
    max_len_text = 500
    max_len_summary = 50
    latent_dim = 500

    # loading
    with open('./src/LSTM_BiLSTM/tokenizerx.pickle', 'rb') as handle:
        x_tokenizer = pickle.load(handle)
    # loading
    with open('./src/LSTM_BiLSTM/tokenizery.pickle', 'rb') as handle:
        y_tokenizer = pickle.load(handle)

    reverse_target_word_index = y_tokenizer.index_word
    reverse_source_word_index = x_tokenizer.index_word
    target_word_index = y_tokenizer.word_index
    x_voc_size = len(x_tokenizer.word_index) + 1
    y_voc_size = len(y_tokenizer.word_index) + 1

    # Declare the mapping of word contraction.
    contraction_mapping = {"ain't": "is not"}
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Build a 3-stacked LSTM for the encoder.

    # Define input and embedding layer.
    encoder_inputs = Input(shape=(max_len_text,))
    enc_emb = Embedding(x_voc_size, latent_dim, trainable=True)(encoder_inputs)

    # LSTM 1.
    encoder_lstm1 = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
    encoder_output1, state_h1, state_c1, *_ = encoder_lstm1(enc_emb)

    # LSTM 2.
    encoder_lstm2 = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
    encoder_output2, state_h2, state_c2, *_ = encoder_lstm2(encoder_output1)

    # LSTM 3.
    encoder_lstm3 = Bidirectional(LSTM(latent_dim, return_state=True, return_sequences=True))
    encoder_outputs, state_h, state_c, *_ = encoder_lstm3(encoder_output2)

    # Build the decoder.

    # Define input and embedding layer.
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(y_voc_size, latent_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    # LSTM in decoder uses encoder_states as initial state.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    # Define attention layer.
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

    # Concat the attention output and the decoder LSTM output.
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    # Define dense layer.
    decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Encoder inference.
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

    # Decoder inference.
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_len_text, 2*latent_dim))

    # Get the embeddings of the decoder sequence.
    dec_emb2 = dec_emb_layer(decoder_inputs)

    # To predict next word in the sequence, set the initial states to states from the previous timestep.
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    # Attention inference.
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # The dense softmax layer to generate probability distribution over the target vocabulary.
    decoder_outputs2 = decoder_dense(decoder_inf_concat)

    # Final decoder model.
    decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c], [decoder_outputs2] + [state_h2, state_c2])

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.load_weights('./src/LSTM_BiLSTM/bilstm_model_weight.h5')
    xx = text_cleaner(text, contraction_mapping, stop_words)
    aa = []
    aa.append(xx)
    aa = x_tokenizer.texts_to_sequences(aa)
    aa = pad_sequences(aa, maxlen=max_len_text, padding='post')
    return decode_sequence(aa.reshape(1, max_len_text), encoder_model, decoder_model, target_word_index, reverse_target_word_index)
