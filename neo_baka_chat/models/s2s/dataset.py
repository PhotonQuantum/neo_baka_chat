from functools import lru_cache
from itertools import zip_longest

import torch
from torch.utils.data import Dataset as _Dataset

from .corpus import Corpus


class Dataset(_Dataset):
    def __init__(self, corpus: Corpus, batch_size: int):
        self.corpus = corpus
        self.vocab = corpus.vocab

        self.inputs = self.corpus.pairs
        self.inputs.sort(key=lambda x: len(x[0]), reverse=True)
        self.batch_size = batch_size

    def zero_padding(self, seqs):
        return list(zip_longest(*seqs, fillvalue=self.vocab.pad_idx))

    def binary_matrix(self, seqs):
        return [[0 if char == self.vocab.pad_idx else 1 for char in seq] for seq in seqs]

    def input_vec(self, seqs):
        idx_batch = [seq + [self.vocab.eos_idx] for seq in seqs]
        lengths = torch.tensor([len(item) for item in idx_batch])
        pad_list = self.zero_padding(idx_batch)
        # noinspection PyArgumentList
        return torch.LongTensor(pad_list), lengths

    def output_vec(self, seqs):
        idx_batch = [seq + [self.vocab.eos_idx] for seq in seqs]
        pad_list = self.zero_padding(idx_batch)
        mask = self.binary_matrix(pad_list)
        max_length = len(pad_list)
        # noinspection PyArgumentList
        return torch.LongTensor(pad_list), torch.BoolTensor(mask), max_length

    @lru_cache(maxsize=None)
    def __getitem__(self, item):
        batch = self.inputs[item * self.batch_size:(item + 1) * self.batch_size]
        inputs, targets = zip(*batch)
        x = self.input_vec(inputs)
        y = self.output_vec(targets)
        return x, y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.inputs) // self.batch_size
