import math
import random
from copy import deepcopy
from functools import partial
from io import BytesIO
from itertools import count
from typing import Callable, Tuple

import torch
from comet_ml import Experiment
from torch import Tensor
from torch import nn
from torch import optim

from neo_baka_chat.results import Result
from .corpus import Corpus
from .dataset import Dataset
from .hparams import HyperParams
from .net import EncoderRNN, LuongAttnDecoderRNN


class Trainer:
    def __init__(self, corpus: Corpus, hparams: HyperParams, cuda: bool = True):
        self.corpus = corpus
        self.vocab = corpus.vocab
        self.vocab_vector_size = len(self.vocab.i2w_table)

        self.embedding = nn.Embedding(self.vocab_vector_size, hparams.hidden_size)
        self.encoder = EncoderRNN(hparams, self.embedding)
        self.decoder = LuongAttnDecoderRNN(hparams, self.vocab_vector_size, self.embedding)

        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.lr = hparams.learning_rate
        self.dec_lr_ratio = hparams.decoder_learning_ratio
        self.epoch = hparams.max_epoch
        self.gradients_norm = hparams.gradients_norm
        self.min_teacher_forcing_ratio = hparams.min_teacher_forcing_ratio

        self.teacher_forcing_ratio = 1

        self.state_h = None
        self.state_c = None
        self.criterion = self.prepare_criterion()

        self.is_cuda = cuda
        self.best_model = None

        random.seed(42)

    def prepare_optimizers(self):
        return optim.Adam(self.encoder.parameters(), lr=self.lr), optim.Adam(self.decoder.parameters(),
                                                                             lr=self.lr * self.dec_lr_ratio)

    def prepare_criterion(self) -> Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, float]]:
        def masked_nllloss(inp: Tensor, target: Tensor, mask: Tensor) -> Tuple[Tensor, float]:
            n_total = mask.sum()
            cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
            loss = cross_entropy.masked_select(mask).mean()
            if self.is_cuda:
                loss = loss.cuda()
            return loss, n_total.item()

        return masked_nllloss

    @staticmethod
    def prepare_teaching_scheduler(max_steps: int, target_ratio: float):
        def exp_sched(_k: int, step: int) -> float:
            return _k / (_k + math.exp(step / _k))

        k: int = 0
        for k in count(1):
            try:
                if exp_sched(k, max_steps) > target_ratio:
                    break
            except OverflowError:
                pass
        return partial(exp_sched, k)

    # noinspection Mypy
    def prepare_dataset(self) -> Dataset:
        return Dataset(self.corpus, self.batch_size)

    def train_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, float]:
        x, y = batch
        x_vec, x_len = x
        y_vec, y_mask, y_max_len = y
        if self.is_cuda:
            x_vec = x_vec.cuda()
            x_len = x_len.cuda()
            y_vec = y_vec.cuda()
            y_mask = y_mask.cuda()

        # forward encoder
        encoder_outputs, encoder_hidden = self.encoder(x_vec, x_len)

        # forward decoder
        # noinspection PyArgumentList, Mypy
        decoder_input = torch.LongTensor([[self.vocab.sos_idx for _ in range(self.batch_size)]])
        if self.is_cuda:
            # noinspection Mypy
            decoder_input = decoder_input.cuda()

        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        loss = 0
        print_losses = []
        n_totals = 0
        # noinspection Mypy
        for t in range(y_max_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            if use_teacher_forcing:
                # noinspection Mypy
                decoder_input = y_vec[t].view(1, -1)
            else:
                _, topi = decoder_output.topk(1)
                # noinspection PyArgumentList,Mypy
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                if self.is_cuda:
                    # noinspection Mypy
                    decoder_input = decoder_input.cuda()
            # noinspection Mypy
            mask_loss, n_total = self.criterion(decoder_output, y_vec[t], y_mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

        return loss, sum(print_losses) / n_totals

    def fit(self, experiment: Experiment, cuda: bool = True) -> Result:
        # Prepare comet.ml logger
        experiment.log_parameters(self.hparams)  # log hyper parameters

        # noinspection PyUnresolvedReferences,Mypy
        torch.backends.cudnn.benchmark = True  # enable cudnn benchmark mode for better performance
        self.is_cuda = cuda
        if self.is_cuda:
            self.encoder.cuda()  # move model to cuda
            self.decoder.cuda()  # move model to cuda

        train_set = self.prepare_dataset()
        self.criterion = self.prepare_criterion()
        max_iters = self.epoch * len(train_set)
        teaching_sched = self.prepare_teaching_scheduler(max_iters, self.min_teacher_forcing_ratio)
        encoder_optim, decoder_optim = self.prepare_optimizers()

        self.encoder.train()  # set the model to train mode
        self.decoder.train()  # set the model to train mode

        iters = 0
        with experiment.train():
            for e in range(self.epoch):
                batch_loss = 0.0
                for batch in train_set:
                    encoder_optim.zero_grad()
                    decoder_optim.zero_grad()

                    iters += 1
                    self.teacher_forcing_ratio = teaching_sched(iters)
                    experiment.log_metric("teacher_forcing_ratio", self.teacher_forcing_ratio, iters)

                    loss, step_loss = self.train_step(batch)

                    batch_loss += step_loss

                    loss.backward()

                    _ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradients_norm)
                    _ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.gradients_norm)

                    encoder_optim.step()
                    decoder_optim.step()

                avg_loss = batch_loss / len(train_set)
                experiment.log_metric("epoch_loss", avg_loss, step=e)

        state_dicts = (deepcopy(self.encoder.state_dict()),
                       deepcopy(self.embedding.state_dict()),
                       deepcopy(self.decoder.state_dict()))
        return Result(avg_loss, e, state_dicts)

    @staticmethod
    def dumps(state_dict):
        tmp_buffer = BytesIO()
        torch.save(state_dict, tmp_buffer)
        tmp_buffer.seek(0)
        return tmp_buffer.read()
