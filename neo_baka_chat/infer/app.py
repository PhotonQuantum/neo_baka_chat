import datetime
import logging
import os
import time
from typing import List, Optional, Union

from fastapi import Body, FastAPI
from pydantic import BaseModel

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


class InferInput(BaseModel):
    sentences: List[str]


class Stat(BaseModel):
    version: datetime.datetime
    net_meta: dict


class InferenceResult(BaseModel):
    response: List[str]
    time: int


class BaseUpdateResult(BaseModel):
    updated: bool
    current_version: datetime.datetime


class UpdateResult(BaseUpdateResult):
    updated = True
    from_version: datetime.datetime


class NoUpdateResult(BaseUpdateResult):
    updated = False


@app.get("/", response_model=Stat)
async def root():
    _version = model.datetime
    return {"version": _version, "net_meta": session.meta}


@app.post("/infer", response_model=InferenceResult)
async def infer(_input: InferInput):
    t_begin = time.time()
    result = session.run(_input.sentences)
    t_end = time.time()

    return {"response": result, "time": int((t_end - t_begin) * 1000)}


# noinspection PyTypeChecker
@app.post("/update", response_model=Union[UpdateResult, NoUpdateResult])
async def update(force: Optional[bool] = False):
    global session, model
    old_version = model.datetime
    logging.warning("Finding latest model.")
    latest_model = bucket.list_models(OSS_PREFIX)[-1]
    if force or latest_model.datetime > model.datetime:
        logging.warning("Updating model.")
        current_version = latest_model.datetime
        session = load_session(model)
        model = latest_model
        logging.warning(f"Model updated from {old_version} to {current_version}")
        return {"updated": True, "current_version": current_version, "from_version": old_version}
    else:
        return {"updated": False, "current_version": old_version}
