import logging
import os
from dataclasses import dataclass
from functools import reduce

from .session import TrainSession
from ..oss import OSS

DISABLE_CUDA = os.environ.get("DISABLE_CUDA") == "true"


@dataclass
class Config:
    model: str
    hparams: dict
    dataset_prefix: str
    save_model_prefix: str


def load_session(config: Config, bucket: OSS) -> TrainSession:
    logging.warning("Downloading dataset.")
    datasets = bucket.list_dataset(config.dataset_prefix)
    dataset = reduce(lambda x, y: x + y, map(lambda x: x.read(), datasets))

    logging.warning(f"Loading net {config.model}.")
    session = TrainSession(dataset, config.hparams, config.model, not DISABLE_CUDA)

    logging.warning("Standby.")
    return session
