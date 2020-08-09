from dataclasses import dataclass
from datetime import datetime
from typing import List, Union


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


def deserialize(data: Union[dict, list]) -> List[Message]:
    if isinstance(data, dict):
        _data = list(data.values())[0]
    elif isinstance(data, list):
        _data = data
    else:
        raise TypeError
    return [Message.loads(entry) for entry in _data]
