import numpy as np
import torch
from torch.utils.data import Dataset as _Dataset

from .corpus import Corpus


class Dataset(_Dataset):
    def __init__(self, corpus: Corpus, seq_size: int):
        self.seq_count = int(len(corpus.sequence) / seq_size)
        self.seq_size = seq_size
        text_data = corpus.sequence[:self.seq_size * self.seq_count]
        shifted_data = []
        shifted_data[:-1] = text_data[1:]
        shifted_data.append(text_data[0])
        self.input_data = torch.from_numpy(np.array(text_data).reshape(self.seq_count, -1))
        self.output_data = torch.from_numpy(np.array(shifted_data).reshape(self.seq_count, -1))

    def __getitem__(self, item):
        return self.input_data[item], self.output_data[item]

    def __len__(self):
        return self.seq_count
