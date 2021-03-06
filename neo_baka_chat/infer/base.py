from abc import ABC, abstractmethod
from typing import List

from neo_baka_chat.hparams import BaseHyperParams
from neo_baka_chat.vocab import Vocab


class AbstractInference(ABC):
    @abstractmethod
    def __init__(self, vocab: Vocab, hparams: BaseHyperParams): ...

    @abstractmethod
    def load_state_dict(self, weights):
        return NotImplemented

    @abstractmethod
    def infer(self, sequence: List[int]) -> List[int]:
        return NotImplemented
