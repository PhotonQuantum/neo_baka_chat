from typing import List


def split_list(seq: list, sep) -> List[list]:
    idx = [-1] + [i for i, x in enumerate(seq) if x == sep] + [len(seq)]
    slices = [slice(x + 1, y) for x, y in zip(idx[:-1], idx[1:])]
    split_seq = [seq[_slice] for _slice in slices]
    return split_seq
