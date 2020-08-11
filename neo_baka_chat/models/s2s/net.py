from typing import Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import Embedding
# noinspection PyPep8Naming
from torch.nn import functional as F

from .hparams import HyperParams


# noinspection PyAbstractClass
class EncoderRNN(nn.Module):
    def __init__(self, hparams: HyperParams, embedding: Embedding):
        super(EncoderRNN, self).__init__()
        self.n_layers = hparams.n_layers
        self.hidden_size = hparams.hidden_size
        self.dropout = hparams.encoder_dropout
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers,
                          dropout=(0 if self.n_layers == 1 else self.dropout), bidirectional=True)

    # noinspection Mypy
    def forward(self, x: Tensor, input_lengths: Tensor, hidden=None) -> Tuple[Tensor, Tensor]:
        # Convert word indexes to embeddings
        embedded = self.embedding(x)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        # noinspection PyTypeChecker
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# noinspection PyAbstractClass
class Attn(nn.Module):
    def __init__(self, hparams: HyperParams):
        super().__init__()
        self.method = hparams.attn_method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hparams.hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            # noinspection PyArgumentList,Mypy
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))

    @staticmethod
    def dot_score(hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        else:
            raise RuntimeError

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# noinspection PyAbstractClass
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, hparams: HyperParams, vocab_vector_size: int, embedding: Embedding):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_method = hparams.attn_method
        self.hidden_size = hparams.hidden_size
        self.output_size = vocab_vector_size
        self.n_layers = hparams.n_layers
        self.dropout = hparams.decoder_dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers,
                          dropout=(0 if self.n_layers == 1 else self.dropout))
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.attn = Attn(hparams)

    # noinspection Mypy
    def forward(self, input_step: Tensor, last_hidden: Tensor, encoder_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
