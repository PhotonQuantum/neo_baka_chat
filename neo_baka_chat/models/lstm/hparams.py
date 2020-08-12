from dataclasses import dataclass

from neo_baka_chat.hparams import BaseHyperParams


@dataclass(frozen=True)
class HyperParams(BaseHyperParams):
    seq_size: int = 64,
    batch_size: int = 128,
    learning_rate: float = 0.005,
    max_epoch: int = 250,
    embed_size: int = 256,
    lstm_size: int = 512,
    lstm_layers: int = 1,
    gradients_norm: float = 5
