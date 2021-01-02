import os
import torch
import numpy as np
import re
import queue
import torch.nn as nn
import random
import os
import sys


def create_emb(weight_matrix, non_trainable=False):
    # create embedding matrix with the pretrained GloVe embedding matrix
    emb_layer = torch.nn.Embedding(weight_matrix.shape[0], weight_matrix.shape[1])
    emb_layer.load_state_dict({'weight': weight_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False  # turn off training for the embedding vector
    return emb_layer


class Encoder(nn.Module):
    def __init__(self, kernel_size, filter_size, dropout, num_hidden, layers, weight_matrix, embed_dim, device):
        super(Encoder, self).__init__()
        self.layers = layers
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.num_hidden = num_hidden
        self.device = device
        self.embed_dim = embed_dim

        # convolutional layers
        self.conv1 = torch.nn.Conv1d(embed_dim, self.filter_size, self.kernel_size[0], stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(embed_dim, self.filter_size, self.kernel_size[1], stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(embed_dim, self.filter_size, self.kernel_size[2], stride=1, padding=2)
        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = create_emb(weight_matrix, True)
        # encoder LSTM
        self.lstm = torch.nn.LSTM(input_size=self.embed_dim, hidden_size=self.num_hidden,
                                  num_layers=self.layers, batch_first=True, dropout=0.5,
                                  bidirectional=True)

    def forward(self, x, hidden):
        x = self.embedding(x)  # embed the input text

        # convolutional layers
        # x1 = torch.tanh(self.dropout(self.conv1(x)))
        # x2 = torch.tanh(self.dropout(self.conv2(x)))
        # x3 = torch.tanh(self.dropout(self.conv3(x)))
        # apply dropout
        lstm_in = self.dropout((x))
        # encoder LSTM
        output, (h_hidden, c_hidden) = self.lstm(lstm_in, hidden)
        # hidden state of the encoder (in case of multilayer LSTM)
        h = torch.cat((h_hidden[-1, :, :], h_hidden[-2, :, :]), 1)
        c = torch.cat((c_hidden[-1, :, :], c_hidden[-2, :, :]), 1)

        hidden = (h.unsqueeze(0), c.unsqueeze(0))

        return output, hidden

    def init_hidden(self, batch_size):
        # initialize the hidden state as the zero tensor
        weight = next(self.parameters()).data
        hidden = (weight.new(self.layers * 2, batch_size, self.num_hidden).zero_(),
                  weight.new(self.layers * 2, batch_size, self.num_hidden).zero_())
        return hidden


# code for the attentional decoder
class AttentionDecoder(nn.Module):
    def __init__(self, num_hidden, dropout, vocab_size, layers, weight_matrix, embed_dims, device):
        super(AttentionDecoder, self).__init__()
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.layers = layers
        self.vocab_size = vocab_size
        self.embed_dims = embed_dims
        self.device = device
        self.dropout_layer = torch.nn.Dropout(self.dropout)

        # Linear layers for the attention mechanism
        self.V = torch.nn.Linear(self.num_hidden, 1)

        # layer to get the pointer probability
        self.generator = torch.nn.Linear(2 * self.num_hidden + self.embed_dims, 1)
        # output layer to vocab
        self.output_layer = torch.nn.Linear(2 * self.num_hidden, self.vocab_size)
        # softmax function
        self.softmax = torch.nn.Softmax(dim=1)
        # decoder LSTM
        self.lstm = torch.nn.LSTM(input_size=self.num_hidden + self.embed_dims, hidden_size=self.num_hidden,
                                  num_layers=self.layers, batch_first=True, dropout=0.5,
                                  bidirectional=False)
        # embedding matrix
        self.embedding = create_emb(weight_matrix, True)
        self.sig = torch.nn.Sigmoid()
        self.device = device

    def forward(self, x, enc_out, hidden, text, batch_size):
        # decoder
        # Decoder Input Shape: [batch_size]
        x = self.embedding(x).unsqueeze(1)
        # Decoder Embedded Shape: [batch_size,1,embed_dim]
        x = self.dropout_layer(x)

        # Bahdanau Attention
        dec_a = hidden[0].permute(1, 0, 2)
        enc_score = self.V(torch.tanh(enc_out + dec_a))  # attention score
        # Attention Score Shape: [batch_size,input_seq_len,1]
        enc_weight = self.softmax(enc_score)  # attention weight
        # Attention Weight Shape: [batch_size,input_seq_len,1]
        enc_context = torch.mul(enc_weight, enc_out)  # find the context vector
        # Attention Context Shape: [batch_size,input_seq_len,decoder_num_hidden]
        enc_context = enc_context.sum(1)
        # Attention Context Shape: [batch_size,decoder_num_hidden]
        enc_context.unsqueeze_(1)
        # Attention Context Shape: [batch_size,1,decoder_num_hidden]

        d_in = torch.cat((x, enc_context), 2)
        # Decoder Input Shape: [batch_size,1,decoder_num_hidden+embed_dims]

        # run the decoder LSTM
        d_output, hidden = self.lstm(d_in, hidden)
        # Decoder Output Shape: [batch_size,1,decoder_num_hidden]
        # Decoder Hidden Shape: [1,batch_size,decoder_num_hidden]

        # concatenate output with the encoder context tensor
        output = torch.cat((d_output.squeeze(1), enc_context.squeeze(1)), 1)
        # Decoder Output Shape: [batch_size,2*decoder_num_hidden]

        output_generator = torch.cat((enc_context.squeeze(1), d_output.squeeze(1), x.squeeze(1)), 1)
        # Generator Input Shape: [batch_size,2*decoder_num_hidden + embed_dims]

        p_gen = self.sig(self.generator(output_generator).squeeze(1))

        # pointer-generator
        p_pointer = 1 - p_gen
        pointer_prob = torch.zeros([batch_size, self.vocab_size], device=self.device)
        for i in range(batch_size):
            pointer_prob[i, text[i, :]] = enc_weight[i, :, 0]  # pointer probability weights are the attention scores
        generator_prob = self.output_layer(output)  # output layer to get vocabulary probability
        output_probability = torch.mul(p_pointer.unsqueeze(1), pointer_prob) + torch.mul(p_gen.unsqueeze(1),
                                                                                         generator_prob)

        return output_probability, hidden

    def init_hidden(self, batch_size):
        # initialize the hidden state as the zero tensor
        weight = next(self.parameters()).data
        hidden = (weight.new(self.layers, batch_size, 2 * self.num_hidden).zero_(),
                  weight.new(self.layers, batch_size, 2 * self.num_hidden).zero_())
        return hidden


# create a beam search node to store the running sequences for the beam search decoder
# and records the hidden state associated with the sequence, the probability and the loss
class BeamNode(object):
    def __init__(self, hidden_state, seq, prob, length, loss):
        self.hidden = hidden_state
        self.seq = seq
        self.prob = prob
        self.len = length
        self.loss = loss
        self.s = self.score()

    def score(self):
        return self.prob / float(self.len - 1 + 1e-6)  # calculates the score of a sequence


class Seq2Seq(nn.Module):
    def __init__(self, kernel_size, filter_size, dropout, num_hidden, enc_layers, weight_embedding,
                 vocab_size, dec_layers, embed_dims, device):
        super(Seq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.encoder = Encoder(kernel_size, filter_size, dropout, num_hidden, enc_layers, weight_embedding, embed_dims,
                               device)
        self.decoder = AttentionDecoder(2 * num_hidden, dropout, vocab_size, dec_layers, weight_embedding, embed_dims,
                                        device)

    def forward(self, x, target, e_hidden, criterion, batch_size):
        # training the decoder
        loss = 0
        prediction = target[:, 0].unsqueeze(1)  # records the running sequences generated by the decoder
        enc_output, enc_hidden = self.encoder(x, e_hidden)  # run the encoder
        d_hidden = enc_hidden  # the decoder input hidden state is the encoder output hidden state
        dec_input = target[:, 0]  # the decoder input starts of as the 'bos' token
        for t in range(1, target.shape[1]):
            # run the decoder
            logits, d_hidden = self.decoder(dec_input, enc_output, d_hidden, x, batch_size)
            dec_input = target[:, t]  # teacher forcing turned on
            loss += criterion(logits, target[:, t])  # calculate the loss function
            # add to get the running prediction output by the decoder
            prediction = torch.cat((prediction, torch.argmax(logits, dim=0).unsqueeze(1)), 0)
        return loss, prediction

    def inference_greedy(self, x, target, e_hidden, criterion, batch_size):
        loss = 0
        prediction = target[:, 0].unsqueeze(1)  # records the running sequences generated by the decoder
        enc_output, enc_hidden = self.encoder(x, e_hidden)  # run the encoder
        d_hidden = enc_hidden  # the decoder input hidden state is the encoder output hidden state
        dec_input = target[:, 0]  # the decoder input starts of as the 'bos' token
        for t in range(1, target.shape[1]):
            # run the decoder
            logits, d_hidden = self.decoder(dec_input, enc_output, d_hidden, x, batch_size)
            # the input to the decoder at the next step is the argument with the largest probability
            dec_input = torch.argmax(logits, dim=1)
            loss += criterion(logits, target[:, t])  # calculate the loss
            # add to get the running prediction output by the decoder
            prediction = torch.cat((prediction, torch.argmax(logits, dim=0).unsqueeze(1)), 0)
        return loss, prediction

    def inference_beam(self, x, target, e_hidden, criterion, beam_width, word2idx, batch_size):
        decoded = []
        losses = 0
        enc_output, enc_hidden = self.encoder(x, e_hidden)  # run the encoder
        for i in range(batch_size):
            # for each sentence in the batch
            prediction = target[i, 0].view([1]).unsqueeze(1)  # running prediction tensor
            dec_hidden = enc_hidden[0].permute(1, 0, 2)[i, :, :].unsqueeze(0)
            dec_input = target[i, 0].view([1])
            d_hidden = (enc_hidden[0][:, i, :].unsqueeze(0), enc_hidden[1][:, i, :].unsqueeze(0))
            first_node = BeamNode(d_hidden, dec_input, 0, 1, 0)  # the first node is the 'bos' token
            nodes = queue.PriorityQueue(maxsize=beam_width)  # create a priority queue to store the beam search nodes
            nodes.put((-first_node.score(), first_node))  # place the first node in the queue
            for t in range(1, target.shape[1]):
                # go through each of the words in the target
                candidatenodes = []  # stores the candidate nodes for a one step look ahead
                candidatescore = []  # stores the candidate scores for a one step look ahead
                donenodes = []  # stores the sequences that have been completed - contain 'eos' token

                # This runs through the 10 sequences in the queue and does a one step look ahead
                # to find 100 possible candidates. The top 10 candidates are then added to the queue
                # and the next word of the decoder is computed

                for _ in range(nodes.qsize()):
                    # for each sequence in the queue
                    sc, nodex = nodes.get()  # get the node from the queue
                    seq = nodex.seq.view([-1, 1])  # the sequence in the node
                    dec_input = seq[-1, :]  # the last word of the sequence is the input to the decoder
                    hidden = nodex.hidden  # the hidden state associated with the sequence
                    d_hidden = (hidden[0], hidden[1])  # the hidden state of the decoder

                    if dec_input == word2idx['eos']:
                        # if the sequence is completed, then it is added to 'donenodes' list
                        donenodes.append((sc, nodex))
                    else:
                        # run the decoder
                        logits, d_hidden = self.decoder(dec_input, enc_output[i, :, :].unsqueeze(0), d_hidden, x,
                                                        batch_size=1)
                        # calculate the loss
                        loss = criterion(logits, target[i, t].view([1]))
                        # pick the top 10 logits values
                        log_p, index = torch.topk(logits, beam_width)
                        for k in range(beam_width):
                            # create a beam node for each of the top 10 to get the candidate nodes
                            node = BeamNode(d_hidden, torch.cat([nodex.seq, index[0, k].unsqueeze(0)]),
                                            nodex.prob + log_p[0][k], nodex.len + 1, nodex.loss + loss)
                            score = -node.score()
                            candidatenodes.append([score, node])
                # put the finished sequences first in the queue first and then fill with candidate nodes
                for score, n in donenodes:
                    nodes.put((score, n))
                for score, n in candidatenodes[nodes.qsize():beam_width]:
                    nodes.put((score, n))
            _, output_node = nodes.get()

            # the output of the beam search decoder is the sequence with the lowest score
            decoded.append(output_node.seq)
            losses += output_node.loss
            del nodes
        return losses / batch_size, decoded


def get_batches(x, y, batch_size=100):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size+1], y[:n_batches*batch_size+1]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


def preprocess(seq):
    seq = seq.lower()  # lower case
    seq = seq.replace('#', '<num>')  # replace the token '#' with '<num>'
    seq = re.sub(">.<", ' point ', seq)  # replace the token '>.<' with ' point '
    seq = re.sub(r"[^a-z?.!,'<>]+", " ", seq)  # replace other tokens with a space
    seq = seq.rstrip().strip()  # strip white space
    seq = 'bos ' + seq + ' eos'  # beginning and end tokens for each sentence
    return seq


def summarize_CNN(input_seq):
    # load in the data
    dirname = os.path.dirname(__file__)
    print("Loading data")
    data = np.load(os.path.join(dirname, 'data.npz'), allow_pickle=True)
    word2idx = data['word2idx']
    word2idx = dict(word2idx.item())
    idx2word = data['idx2word']
    idx2word = dict(idx2word.item())
    print('Data Loaded!')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device to run the code on GPU
    device = torch.device("cpu")

    # hyperparameters of the mode
    batch_size = 32
    model = torch.load(os.path.join(dirname, "model_cpu.pkl"))
    # cross entropy loss function with the padding ignored
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    beam_width = 10  # beam search width is 10

    input_seq = preprocess(input_seq)
    sentence = []
    for word in input_seq.split():
        if word not in word2idx:
            word2idx[word] = random.choice(np.arange(0, len(word2idx)))
        sentence.extend([word2idx[word]])
    text_len = 50
    text_feature = np.zeros(text_len, dtype=int)
    text_feature = np.array(sentence)[:text_len]
    text_feature = np.array([text_feature])
    summary_len = 30
    summary_feature = np.zeros(summary_len, dtype=int)
    summary_feature = np.array([summary_feature])

    # beam search decoder
    with torch.no_grad():
        loss_beam = []
        beam_predict = []  # save beam search decoder outputs
        summary_validation = []  # save validation target summaries
        text_validation = []  # save validation input text
        model.eval()
        # initialize the encoder hidden state
        val_hidden = model.encoder.init_hidden(batch_size=1)
        for x_val, y_val in get_batches(text_feature, summary_feature, batch_size):
            # convert data to PyTorch tensor
            x_val = torch.from_numpy(x_val).to(device).long()
            y_val = torch.from_numpy(y_val).to(device).long()
            val_hidden = tuple([each.data for each in val_hidden])
            # run the beam search decoder
            val_loss, prediction = model.inference_beam(x_val, y_val, val_hidden, criterion, beam_width, word2idx, batch_size=1)
            loss_beam.append(val_loss.item())
            beam_predict.append(prediction)
            summary_validation.append(y_val)
            text_validation.append(x_val)

        beam = []
        for i in beam_predict:
            for p in i:
                array = torch.Tensor.cpu(p).numpy()
                beam_string = ''
                for q in array:
                    beam_string += idx2word[q] + ' '
                beam.append(beam_string)
        result = beam[0].split(' ', 1)[1]
    return result


sys.path.append(__file__)
# text = 'test string'
# print(summarize_CNN(text))
