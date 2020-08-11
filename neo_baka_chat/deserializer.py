import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterator, List, Optional, Set, Tuple, Union

import ftfy

# noinspection RegExpRedundantEscape
sub_list = [
    (re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'), ""),
    (re.compile(r'@.*?;'), ";"),
    (re.compile(r'@.*? '), ""),
    (re.compile(r'\[(图片|表情|短视频)\]'), ""),
    (re.compile(r'.*撤回了一条消息'), ""),
    (re.compile(r'�'), "")
]


@dataclass(frozen=True)
class Message:
    name: str
    id: int
    time: datetime
    text: str
    extra: dict

    @classmethod
    def loads(cls, msg: dict) -> "Message":
        load_msg = msg.copy()
        load_msg["time"] = datetime.fromtimestamp(load_msg["time"])
        return cls(**load_msg)

    def dumps(self) -> dict:
        dump_msg = asdict(self)
        dump_msg["time"] = int(dump_msg["time"].timestamp())
        return dump_msg


def clean_text(messages: Union[List[Message], Message], nicknames: Optional[Set[str]] = None) \
        -> Tuple[Union[List[Message], Message], Set[str]]:
    def _clean(message: Message) -> Message:
        text = message.text
        for pattern, sub in sub_list:
            text = re.sub(pattern, sub, text)  # clean the text with predefined regex patterns
        for name in _nicknames:
            text = text.replace(f"@{name}", "")  # now we remove all nickname.
        text = ftfy.fix_text(text).strip()  # normalize the text
        return Message(message.name, message.id, message.time, text, message.extra)

    def neither_empty_nor_arabic(msg: Message) -> bool:
        return bool(msg.text) and not bool(msg.extra["arabic"])

    _messages: Iterator[Message]
    if isinstance(messages, Message):
        _nicknames = set(messages.name).union(set() if nicknames is None else nicknames)
        return _clean(messages), _nicknames
    elif isinstance(messages, list):
        _nicknames: Set[str] = set(message.name for message in messages).union(
            set() if nicknames is None else nicknames)
        _messages = map(_clean, messages)
        _messages = filter(neither_empty_nor_arabic, _messages)
        return list(_messages), _nicknames
    else:
        raise TypeError


def deserialize(data: Union[dict, list]) -> List[Message]:
    if isinstance(data, dict):
        _data = list(data.values())[0]
    elif isinstance(data, list):
        _data = data
    else:
        raise TypeError
    return list(map(Message.loads, _data))
