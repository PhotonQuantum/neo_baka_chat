from abc import ABC
from typing import List

from neo_baka_chat.hparams import BaseHyperParams
from neo_baka_chat.vocab import Vocab


class AbstractInference(ABC):
    def __init__(self, vocab: Vocab, hparams: BaseHyperParams): ...

    def load_state_dict(self, weights):
        return NotImplemented

    def infer(self, sequence: List[int]) -> List[int]:
        return NotImplemented
