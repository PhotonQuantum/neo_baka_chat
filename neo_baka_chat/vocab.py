from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Union


@dataclass(frozen=True)
class Vocab:
    sep_idx: int
    sos_idx: int
    pad_idx: int
    eos_idx: int
    nicknames: Set[str]
    i2w_table: Dict[int, str]
    w2i_table: Dict[str, int]

    def int_to_word(self, idx: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(idx, int):
            return self.i2w_table[idx]
        elif isinstance(idx, list):
            return [self.i2w_table[_idx] for _idx in idx]
        else:
            raise TypeError

    def word_to_int(self, word: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(word, str):
            return self.w2i_table[word]
        elif isinstance(word, list):
            return [self.w2i_table[_word] for _word in word]
        else:
            raise TypeError

    @classmethod
    def build(cls, words: List[str], nicknames: Optional[Set[str]] = None,
              sep_token: str = "SEP", sos_token: str = "SOS", pad_token: str = "PAD", eos_token: str = "EOS") \
            -> "Vocab":
        nicknames = set() if nicknames is None else nicknames
        word_list = words + [sep_token, sos_token, pad_token, eos_token]
        word_counter = Counter(word_list)
        sorted_word_list = sorted(word_counter, key=word_counter.get, reverse=True)
        i2w_table = {i: w for i, w in enumerate(sorted_word_list)}
        w2i_table = {w: i for i, w in i2w_table.items()}
        sep_idx, sos_idx = w2i_table[sep_token], w2i_table[sos_token]
        pad_idx, eos_idx = w2i_table[pad_token], w2i_table[eos_token]
        return cls(sep_idx, sos_idx, pad_idx, eos_idx, nicknames, i2w_table, w2i_table)

    @classmethod
    def loads(cls, data: dict) -> "Vocab":
        return cls(**data)

    def dumps(self) -> dict:
        return asdict(self)
