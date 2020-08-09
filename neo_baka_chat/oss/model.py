import json
import time
from dataclasses import dataclass
from datetime import datetime
from os import path
from typing import List, Optional, Union

from oss2 import Bucket, ObjectIterator

from .base import BaseOSS


@dataclass(frozen=True)
class ModelObject:
    path: str
    key: str
    _bucket: Bucket

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(int(self.key.split("_")[1]))

    def list(self) -> List[str]:
        return [entry.key.split("/")[-1] for entry in ObjectIterator(self._bucket, path.join(self.path, self.key))]

    def get(self, key: str, is_json: bool = False) -> Union[dict, bytes]:
        url = path.join(self.path, self.key, key)
        if is_json:
            return json.load(self._bucket.get_object(url))
        else:
            return self._bucket.get_object(url).read()

    def put(self, key: str, data: Union[bytes, dict]):
        url = path.join(self.path, self.key, key)
        if isinstance(data, bytes):
            _data = data
        elif isinstance(data, (list, dict)):
            _data = json.dumps(data)
        else:
            raise TypeError
        self._bucket.put_object(url, _data)

    def delete(self, key: Optional[str] = None):
        if key:
            self._bucket.delete_object(path.join(self.path, self.key, key))
            return
        delete_list = [file.key for file in ObjectIterator(self._bucket, path.join(self.path, self.key))] + [
            path.join(self.path, self.key)]
        self._bucket.batch_delete_objects(delete_list)


class ModelMixin(BaseOSS):
    def __init__(self):
        super().__init__()

    def list_models(self, prefix: str) -> List[ModelObject]:
        models = [entry.key for entry in ObjectIterator(self.bucket, prefix=prefix)]
        if not models:
            return []

        models = [entry.split("/")[1] for entry in models]  # 2020-03-12_1584028278
        models.sort(key=lambda x: x.split("_")[1])

        return [ModelObject(prefix, model, self.bucket) for model in models]

    def rotate_models(self, prefix: str, keep_models: int = 3):
        models = self.list_models(prefix)
        for model in models[:-keep_models]:
            model.delete()

    def create_model(self, prefix: str) -> ModelObject:
        ts = int(time.time())
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        key = f"{dt}_{ts}"
        return ModelObject(prefix, key, self.bucket)
