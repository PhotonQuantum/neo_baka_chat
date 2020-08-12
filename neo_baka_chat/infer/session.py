import importlib
from functools import reduce
from typing import List

from neo_baka_chat.hparams import BaseHyperParams
from neo_baka_chat.utils import split_list
from neo_baka_chat.vocab import Vocab
from .base import AbstractInference


class InferenceSession:
    # noinspection PyUnresolvedReferences
    def __init__(self, vocab: Vocab, meta: dict, model: str, weights):
        self.vocab = vocab
        self.meta = meta

        self.hparams_module = importlib.import_module(".hparams", f"neo_baka_chat.models.{model}")
        self.infer_module = importlib.import_module(".infer", f"neo_baka_chat.models.{model}")
        self.hparams: BaseHyperParams = self.hparams_module.HyperParams.loads(self.meta)
        self.inference: AbstractInference = self.infer_module.Inference(self.vocab, self.hparams)
        self.inference.load_state_dict(weights)

    def run(self, sentences: List[str]) -> List[str]:
        split_sentences = map(self._match_vocab, sentences)
        sequences = map(self.vocab.word_to_int, split_sentences)
        flatten_seq = reduce(lambda x, y: x + [self.vocab.sep_idx] + y, sequences)

        result_seq = self.inference.infer(flatten_seq)
        result_seqs = split_list(result_seq, self.vocab.sep_idx)
        result_sentences = map("".join, map(self.vocab.int_to_word, result_seqs))
        return list(result_sentences)

    def _match_vocab(self, sentence: str) -> List[str]:
        vocab = list(self.vocab.w2i_table.keys())
        vocab.sort(key=len, reverse=True)
        output = []
        while sentence:
            flag = False
            for entry in vocab:
                try:
                    if sentence[:len(entry)] == entry:
                        output.append(entry)
                        sentence = sentence[len(entry):]
                        flag = True
                        break
                except IndexError:
                    continue
            if not flag:
                sentence = sentence[1:]
        return output
