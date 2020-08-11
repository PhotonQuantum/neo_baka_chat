from dataclasses import dataclass

from neo_baka_chat.hparams import BaseHyperParams


@dataclass(frozen=True)
class HyperParams(BaseHyperParams):
    batch_size: int = 128
    learning_rate: float = 0.0001
    decoder_learning_ratio: float = 5.0
    max_epoch: int = 150
    hidden_size: int = 500
    n_layers: int = 2
    attn_method: str = "dot"
    encoder_dropout: float = 0.1
    decoder_dropout: float = 0.1
    teacher_forcing_ratio: float = 0.8
    min_teacher_forcing_ratio: float = 0.01
    gradients_norm: float = 5
