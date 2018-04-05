"""Neural machine translation model implementation."""

from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embedding_dim,
                 _bidirectional,
                 _dropout,
                 num_layers,
                 mode):
        """Initializes the encoder.

        Args:
            src_vocab_size: Number of words in source vocabulary.
            embedding_dim: Size of word embeddings.
        """
        # Always do this when defining a Module.
        super(Encoder, self).__init__()

        # Track input parameters.
        self.bidirectional = _bidirectional
        self.src_vocab_size = src_vocab_size
        self.embedding_dim = embedding_dim
        self._num_layers   = num_layers


        # Define the model layers.

        self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.islstm = False
        if mode == "LSTM":
            self.islstm = True
            self.rnn = nn.LSTM(embedding_dim, embedding_dim, num_layers=num_layers,
                               bidirectional=_bidirectional, dropout=_dropout)
        elif mode == "GRU":
            self.rnn = nn.GRU(embedding_dim, embedding_dim, num_layers=num_layers,
                           bidirectional= _bidirectional,dropout=_dropout)
        else:
            self.rnn = nn.RNN(embedding_dim, embedding_dim)

        # Initialize the embedding weights.
        nn.init.xavier_uniform(self.embedding.weight)

    def forward(self, src, hidden=None):
        """Computes the forward pass of the encoder.

        Args:
            src: LongTensor w/ dimension [seq_len].
            hidden: FloatTensor w/ dimension [embedding_dim].

        Returns:
            net: FloatTensor w/ dimension [seq_len, embedding_dim].
                The encoder outputs for each element in src.
            hidden: FloatTensor w/ dimension [embedding_dim]. The final hidden
                activations.
        """
        # If no initial hidden state provided, then use all zeros.
        if hidden is None:
            if self.bidirectional:
                num_directions = 2
            else:
                num_directions = 1
            hidden = Variable(torch.zeros(self._num_layers  * num_directions, 1, self.embedding_dim))
            if self.islstm:
                c = Variable(torch.zeros(self._num_layers  * num_directions, 1, self.embedding_dim))
                if torch.cuda.is_available():
                    c = c.cuda()
                    hidden = hidden.cuda()
                    hidden = (hidden,c)
                else:
                    hidden = (hidden, c)
            else:
                if torch.cuda.is_available():
                    hidden = hidden.cuda()

        # Forward pass.
        net = self.embedding(src).view(1, 1, -1) # Embed

        net, hidden = self.rnn(net, hidden) # Feed through RNN

        return net, hidden


class Decoder(nn.Module):
    def __init__(self,
                 tgt_vocab_size,
                 embedding_dim,
                 _bidirectional,
                 _dropout,
                 num_layers,
                 mode):
        """Initializes the decoder.

        Args:
            tgt_vocab_size: Number of words in target vocabulary.
            embedding_dim: Size of word embeddings.
        """
        # Always do this when defining a Module.
        super(Decoder, self).__init__()

        # Track input parameters.
        self.tgt_vocab_size = tgt_vocab_size
        self.bidirectional = _bidirectional
        self.embedding_dim = embedding_dim
        self._num_layers  = num_layers

        # Define the model layers.
        self.embedding = nn.Embedding(tgt_vocab_size, embedding_dim)

        if mode == "LSTM":
            self.islstm = True
            self.rnn = nn.LSTM(embedding_dim, embedding_dim, num_layers=num_layers,
                               bidirectional=_bidirectional, dropout=_dropout)
        elif mode == "GRU":
            self.rnn = nn.GRU(embedding_dim, embedding_dim, num_layers=num_layers,
                           bidirectional= _bidirectional,dropout=_dropout)
        else:
            self.rnn = nn.RNN(embedding_dim, embedding_dim)

        if _bidirectional:
            self.fc = nn.Linear(2 * embedding_dim, tgt_vocab_size)
        else:
            self.fc = nn.Linear(embedding_dim, tgt_vocab_size)

        #self.fc = nn.Linear(embedding_dim, tgt_vocab_size)

        # Initialize the embedding weights.
        nn.init.xavier_uniform(self.embedding.weight)

    def forward(self, tgt, hidden):
        """Computes the forward pass of the decoder.

        Args:
            tgt: LongTensor w/ dimension [seq_len, batch_size].
            lengths: LongTensor w/ dimension [seq_len]. Contains the lengths of
                the sequences in tgt.
            hidden: FloatTensor w/ dimension [TBD].

        Returns:
            net: FloatTensor w/ dimension [seq_len, batch_size, embedding_dim].
                The decoder outputs for each element in tgt.
            hidden: FloatTensor w/ dimension [TBD]. The final hidden
                activations.
        """
        # Forward pass.
        net = self.embedding(tgt).view(1, 1, -1) # Embed
        net, hidden = self.rnn(net, hidden) # Feed through RNN
        net = self.fc(net) # Feed through fully connected layer
        net = F.log_softmax(net[0], dim=-1) # Transform to log-probabilities

        return net, hidden

