import logging
import os

import yaml
from comet_ml import Experiment

from neo_baka_chat.oss import OSS
from neo_baka_chat.train.loader import Config, load_session

ALIYUN_ACCESSKEY_ID = os.environ["ALIYUN_ACCESSKEY_ID"]
ALIYUN_ACCESSKEY_SECRET = os.environ["ALIYUN_ACCESSKEY_SECRET"]
ALIYUN_REGION = os.environ["ALIYUN_REGION"]
OSS_BUCKET = os.environ["OSS_BUCKET"]
COMET_KEY = os.environ.get("COMET_KEY")
COMET_PROJECT = os.environ.get("COMET_PROJECT")
CONFIG_PATH = os.environ["CONFIG_PATH"]


def main():
    logging.warning("Connecting to OSS.")
    bucket = OSS(ALIYUN_ACCESSKEY_ID, ALIYUN_ACCESSKEY_SECRET, ALIYUN_REGION, OSS_BUCKET)

    logging.warning("Fetching configuration.")
    _config = yaml.safe_load(bucket.loads(CONFIG_PATH, is_json=False))
    config = Config(**_config)

    session = load_session(config, bucket)

    logging.warning("Training.")
    experiment = Experiment(COMET_KEY, project_name=COMET_PROJECT) if COMET_KEY else None
    result = session.train(experiment, config.mixed_precision)

    logging.warning("Uploading model.")
    meta = session.hparams.dumps()
    meta.update({"epoch": result.epoch, "loss": result.loss, "mixed_precision": result.mixed_precision})
    model = bucket.create_model(config.save_model_prefix)
    model.put("vocab.json", session.corpus.vocab.dumps())
    model.put("meta.json", meta)
    model.put("weights.pth", result.dump_state())

    logging.warning("Complete.")

main()