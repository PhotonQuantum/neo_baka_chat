from abc import ABC, abstractmethod
from typing import List, Optional

from comet_ml import Experiment

from neo_baka_chat.deserializer import Message
from neo_baka_chat.hparams import BaseHyperParams
from neo_baka_chat.results import Result
from neo_baka_chat.vocab import Vocab


class AbstractCorpus(ABC):
    vocab: Vocab

    @classmethod
    @abstractmethod
    def build(cls, messages: List[Message]):
        return NotImplemented


class AbstractTrainer(ABC):
    @abstractmethod
    def __init__(self, corpus: AbstractCorpus, hparams: BaseHyperParams, cuda: bool = True): ...

    @abstractmethod
    def fit(self, experiment: Optional[Experiment] = None) -> Result:
        return NotImplemented
