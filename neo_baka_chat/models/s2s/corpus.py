from dataclasses import dataclass
from typing import List, Tuple

import jieba

from neo_baka_chat.deserializer import Message, clean_text
from neo_baka_chat.utils import split_list
from neo_baka_chat.vocab import Vocab


@dataclass(frozen=True)
class Corpus:
    vocab: Vocab
    pairs: List[Tuple[List[int], List[int]]]

    @classmethod
    def build(cls, messages: List[Message]):
        def suitable_length(msg: Message) -> bool:
            text = msg.text
            return 2 < len(text) < 25

        def get_message_pairs(msgs: List[Message]) -> List[Tuple[Message, Message]]:
            different_person = lambda x: x[0].id != x[1].id
            close_in_time = lambda x: (x[0].time - x[1].time).total_seconds() < 120

            msg_pairs = zip(msgs[:-1], msgs[1:])
            msg_pairs = filter(different_person, msg_pairs)
            msg_pairs = filter(close_in_time, msg_pairs)
            return list(msg_pairs)

        def get_sentence_pairs(msg_pairs: List[Tuple[Message, Message]]) -> List[Tuple[str, str]]:
            return [(msg_1.text, msg_2.text) for msg_1, msg_2 in msg_pairs]

        def flatten(sentence_pairs: List[Tuple[str, str]]) -> str:
            sentences = ["SEP".join(pair) for pair in sentence_pairs]
            return "SEP".join(sentences)

        def unflatten(seq: List[int], sep: int) -> List[Tuple[List[int], List[int]]]:
            sentences = split_list(seq, sep)
            pairs = [(x, y) for x, y in zip(sentences[::2], sentences[1::2])]
            return pairs

        def split_words(sentence: str) -> List[str]:
            jieba.add_word("SEP")
            return jieba.lcut(sentence)

        _messages, nicknames = clean_text(messages)
        _messages = filter(suitable_length, _messages)
        message_pairs = get_message_pairs(list(_messages))
        sentence_pairs = get_sentence_pairs(message_pairs)
        flatten_sentence = flatten(sentence_pairs)

        flatten_words = split_words(flatten_sentence)
        vocab = Vocab.build(flatten_words, nicknames)
        flatten_int = vocab.word_to_int(flatten_words)
        int_pairs = unflatten(flatten_int, vocab.sep_idx)

        return cls(vocab, int_pairs)
