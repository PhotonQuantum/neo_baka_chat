import logging
import os
import time
from typing import List, Optional, Union

from fastapi import FastAPI

from neo_baka_chat.oss import OSS
from .loader import load_session

logging.warning("Standby.")

app = FastAPI()

ALIYUN_ACCESSKEY_ID = os.environ["ALIYUN_ACCESSKEY_ID"]
ALIYUN_ACCESSKEY_SECRET = os.environ["ALIYUN_ACCESSKEY_SECRET"]
ALIYUN_REGION = os.environ["ALIYUN_REGION"]
OSS_BUCKET = os.environ["OSS_BUCKET"]
OSS_PREFIX = os.environ["OSS_PREFIX"]

logging.warning("Connecting to OSS.")
bucket = OSS(ALIYUN_ACCESSKEY_ID, ALIYUN_ACCESSKEY_SECRET, ALIYUN_REGION, OSS_BUCKET)

logging.warning("Finding latest model.")
model = bucket.list_models(OSS_PREFIX)[-1]

session = load_session(model)


@app.get("/")
async def root():
    _version = model.datetime.strftime("%Y-%m-%d %H:%M:%S")
    return {"version": _version, "net_meta": session.meta}


@app.get("/infer")
async def infer(sentences: Union[str, List[str]]):
    sentences = sentences if isinstance(sentences, list) else [sentences]
    t_begin = time.time()
    result = session.run(sentences)
    t_end = time.time()

    return {"response": result, "time": int((t_end - t_begin) * 1000)}


@app.post("/update")
async def update(force: Optional[bool] = False):
    global session, model
    old_version = model.datetime.strftime("%Y-%m-%d %H:%M:%S")
    logging.warning("Finding latest model.")
    latest_model = bucket.list_models(OSS_PREFIX)[-1]
    if force or latest_model.datetime > model.datetime:
        logging.warning("Updating model.")
        current_version = latest_model.datetime.strftime("%Y-%m-%d %H:%M:%S")
        session = load_session(model)
        model = latest_model
        logging.warning(f"Model updated from {old_version} to {current_version}")
        return {"updated": True, "current_version": current_version, "from_version": old_version}
    else:
        return {"updated": False, "current_version": old_version}
