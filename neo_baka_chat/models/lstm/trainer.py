from contextlib import nullcontext
from copy import deepcopy
from typing import Optional, Tuple

import torch
from comet_ml import Experiment
from torch import Tensor
from torch import nn
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from neo_baka_chat.results import Result
from .corpus import Corpus
from .dataset import Dataset
from .hparams import HyperParams
from .net import RNNModule
from ...train.base import AbstractTrainer


class Trainer(AbstractTrainer):
    def __init__(self, corpus: Corpus, hparams: HyperParams, cuda: bool = True):
        super().__init__(corpus, hparams, cuda)
        self.corpus = corpus

        self.hparams = hparams
        self.seq_size = hparams.seq_size
        self.lr = hparams.learning_rate
        self.epoch = hparams.max_epoch
        self.gradients_norm = hparams.gradients_norm
        self.batch_size = hparams.batch_size
        self.vocab_vector_size = len(corpus.vocab.i2w_table)

        self.net = RNNModule(self.vocab_vector_size, hparams)

        self.state_h = None
        self.state_c = None
        self.criterion = self.prepare_criterion()

        self.is_cuda = cuda
        self.best_model = None

    def prepare_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=self.lr)

    @staticmethod
    def prepare_criterion():
        return nn.CrossEntropyLoss()

    def prepare_dataset(self) -> DataLoader:
        return DataLoader(Dataset(self.corpus, self.seq_size), batch_size=self.batch_size,
                          pin_memory=True, drop_last=True)

    def before_train_epoch(self):
        self.state_h, self.state_c = self.net.zero_state()  # reset hidden layers
        if self.is_cuda:
            self.state_h = self.state_h.cuda()
            self.state_c = self.state_c.cuda()

    def train_step(self, batch: Tuple[Tensor, Tensor]):
        x, y = batch
        if self.is_cuda:
            x, y = x.cuda(), y.cuda()

        # forward and calculate loss
        logits, (self.state_h, self.state_c) = self.net.forward(x, (self.state_h, self.state_c))
        loss = self.criterion(logits.transpose(1, 2), y)

        # detach hidden layers
        self.state_h, self.state_c = self.state_h.detach(), self.state_c.detach()

        return loss

    def fit(self, experiment: Optional[Experiment] = None, mixed_precision: bool = True) -> Result:
        if experiment:
            # Prepare comet.ml logger
            experiment.log_parameters(self.hparams)  # log hyper parameters

        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = True  # enable cudnn benchmark mode for better performance
        if self.is_cuda:
            self.net.cuda()  # move model to cuda

        train_set = self.prepare_dataset()
        optimizer = self.prepare_optimizers()

        best_model = Result(999, 0, mixed_precision, None)

        self.net.train()  # set the model to train mode

        scaler = GradScaler() if mixed_precision and self.is_cuda else None

        with experiment.train() if experiment else nullcontext():
            for e in range(self.epoch):
                self.before_train_epoch()

                batch_loss = 0
                for batch_idx, batch in enumerate(train_set):
                    optimizer.zero_grad()

                    with autocast() if scaler else nullcontext():
                        loss = self.train_step(batch)

                    batch_loss += loss.item()

                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    _ = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.gradients_norm)

                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                avg_loss = batch_loss / len(train_set)
                if experiment:
                    experiment.log_metric("epoch_loss", avg_loss, step=e)
                # noinspection PyUnboundLocalVariable
                if avg_loss < best_model.loss:
                    best_model = Result(avg_loss, e, mixed_precision, deepcopy(self.net.state_dict()))

        return best_model
