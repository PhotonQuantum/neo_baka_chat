from dataclasses import dataclass
from io import BytesIO
from typing import Any

import torch


@dataclass(frozen=True)
class Result:
    loss: float
    epoch: int
    state_dicts: Any

    def dump_state(self) -> bytes:
        buffer = BytesIO()
        torch.save(self.state_dicts, buffer)
        buffer.seek(0)
        return buffer.read()
