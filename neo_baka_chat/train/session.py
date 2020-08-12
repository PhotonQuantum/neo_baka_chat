import importlib
from typing import List, Optional

from comet_ml import Experiment

from neo_baka_chat.deserializer import Message
from neo_baka_chat.hparams import BaseHyperParams
from neo_baka_chat.results import Result
from neo_baka_chat.train.base import AbstractCorpus, AbstractTrainer


class TrainSession:
    # noinspection PyUnresolvedReferences
    def __init__(self, messages: List[Message], hparams: dict, model: str, cuda: bool = True):
        self.cuda: bool = cuda

        self.hparams_module = importlib.import_module(".hparams", f"neo_baka_chat.models.{model}")
        self.corpus_module = importlib.import_module(".corpus", f"neo_baka_chat.models.{model}")
        self.trainer_module = importlib.import_module(".trainer", f"neo_baka_chat.models.{model}")

        self.hparams: BaseHyperParams = self.hparams_module.HyperParams.loads(hparams)
        self.corpus: AbstractCorpus = self.corpus_module.Corpus.build(messages)
        self.trainer: AbstractTrainer = self.trainer_module.Trainer(self.corpus, self.hparams, cuda)

    def train(self, experiment: Optional[Experiment] = None) -> Result:
        return self.trainer.fit(experiment)
