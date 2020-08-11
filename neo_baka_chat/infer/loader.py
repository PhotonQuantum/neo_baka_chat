import logging
import os
from io import BytesIO

import torch

from neo_baka_chat.oss.model import ModelObject
from neo_baka_chat.vocab import Vocab
from .session import InferenceSession

MODEL = os.environ["MODEL"]


def load_session(model: ModelObject) -> InferenceSession:
    logging.warning("Fetching weights and parameters.")
    _vocab = model.get("vocab.json", is_json=True)
    meta = model.get("meta.json", is_json=True)
    _weights = model.get("weights.pth", is_json=False)

    logging.warning(f"Loading net {MODEL}.")
    vocab = Vocab.loads(_vocab)
    weight_io = BytesIO(_weights)
    weight_io.seek(0)
    weights = torch.load(weight_io, map_location=torch.device("cpu"))
    session = InferenceSession(vocab, meta, MODEL, weights)

    logging.warning("Standby.")
    return session
