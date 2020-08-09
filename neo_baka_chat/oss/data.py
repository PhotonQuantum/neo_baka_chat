import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from os import path
from typing import List

from oss2 import Bucket, ObjectIterator

from .base import BaseOSS


@dataclass(frozen=True)
class DataObject:
    path: str
    key: str
    _bucket: Bucket

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(int(self.key))

    def read(self) -> dict:
        url = path.join(self.path, self.key)
        extension = self.key.split(".")[-1]
        file = self._bucket.get_object(url)
        if extension == "pickle":
            return pickle.loads(file.read())
        elif extension == "json":
            return json.load(file)
        else:
            raise KeyError


class DataMixin(BaseOSS):
    def __init__(self):
        super().__init__()

    def list_dataset(self, prefix: str) -> List[DataObject]:
        datasets = [entry.key for entry in ObjectIterator(self.bucket, prefix=prefix)]
        if not datasets:
            return []

        models = [entry.split("/")[1] for entry in datasets]
        models.sort()

        return [DataObject(prefix, model, self.bucket) for model in models]

    def put_dataset(self, prefix: str, key: str, data) -> DataObject:
        url = path.join(prefix, key)
        if isinstance(data, (list, dict)):
            self.bucket.put_object(url, json.dumps(data))
        elif isinstance(data, bytes):
            self.bucket.put_object(url, data)
        else:
            self.bucket.put_object(url, pickle.dumps(data))
        return DataObject(prefix, key, self.bucket)
