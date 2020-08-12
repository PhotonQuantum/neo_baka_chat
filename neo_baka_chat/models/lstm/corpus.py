from dataclasses import dataclass
from typing import List

import jieba

from neo_baka_chat.deserializer import Message, clean_text
from neo_baka_chat.train.base import AbstractCorpus
from neo_baka_chat.vocab import Vocab


@dataclass(frozen=True)
class Corpus(AbstractCorpus):
    vocab: Vocab
    sequence: List[int]

    @classmethod
    def build(cls, messages: List[Message]):
        def split_words(sentence: str) -> List[str]:
            jieba.add_word("SEP")
            return jieba.lcut(sentence)

        _messages, nicknames = clean_text(messages)
        plain_messages = [message.text for message in _messages]

        words = split_words("SEP".join(plain_messages))
        vocab = Vocab.build(words, nicknames)
        seq = vocab.word_to_int(words)

        return cls(vocab, seq)
