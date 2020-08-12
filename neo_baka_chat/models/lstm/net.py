from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from .hparams import HyperParams


# noinspection PyAbstractClass
class RNNModule(nn.Module):
    def __init__(self, vocab_vector_size: int, hparams: HyperParams):
        super().__init__()

        self.hparams = hparams
        self.lstm_size = hparams.lstm_size
        self.embedding_size = hparams.embed_size
        self.num_layers = hparams.lstm_layers
        self.batch_size = hparams.batch_size
        self.vocab_vector_size = vocab_vector_size

        self.embedding = nn.Embedding(self.vocab_vector_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size,
                            self.lstm_size,
                            self.num_layers,
                            batch_first=True)
        self.dense = nn.Linear(self.lstm_size, self.vocab_vector_size)

    def forward(self, x: Tensor, prev_state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(self.num_layers, self.batch_size, self.lstm_size),
                torch.zeros(self.num_layers, self.batch_size, self.lstm_size))
