from functools import reduce
from typing import List

import torch

from neo_baka_chat.infer.base import AbstractInference
from neo_baka_chat.vocab import Vocab
from .hparams import HyperParams
from .net import RNNModule


class Inference(AbstractInference):
    def __init__(self, vocab: Vocab, hparams: HyperParams):
        super().__init__(vocab, hparams)
        self.vocab = vocab

        _hparams = hparams.dumps()
        _hparams["batch_size"] = 1
        self.hparams = HyperParams.loads(_hparams)
        self.vocab_vector_size = len(vocab.i2w_table)

        self.net = RNNModule(self.vocab_vector_size, self.hparams)

        self.n = 1
        self.temp = 1

    def _inference(self, sequence: List[int]) -> List[List[int]]:
        """
        Input: A sequence of word vectors, split with sep
        Output: Multiple sequences of word vectors. Each sequence represents a complete sentence.
        """
        self.net.eval()
        with torch.no_grad():
            state_h, state_c = self.net.zero_state()
            for w in sequence[:-1]:
                ix = torch.tensor([[w]]).cpu()
                output, (state_h, state_c) = self.net(ix, (state_h, state_c))

            choice = sequence[-1]
            result = []
            temp_seq = []
            counter = 0
            while counter < self.n:
                ix = torch.tensor([[choice]]).cpu()
                output, (state_h, state_c) = self.net(ix, (state_h, state_c))

                probability = output[0][0].div(self.temp).exp()
                choice = int(torch.multinomial(probability, 1)[0])

                if choice == self.vocab.sep_idx:
                    result.append(temp_seq.copy())
                    temp_seq.clear()
                    counter += 1
                else:
                    temp_seq.append(choice)

        return result

    def infer(self, sequence: List[int]) -> List[int]:
        result = self._inference(sequence + [self.vocab.sep_idx])
        return reduce(lambda x, y: x + [self.vocab.sep_idx] + y, result)

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
