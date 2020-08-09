import urllib.request
from urllib.error import HTTPError, URLError

from oss2 import Auth, Bucket

from .base import BaseOSS
from .data import DataMixin
from .model import ModelMixin


class OSS(ModelMixin, DataMixin, BaseOSS):
    @staticmethod
    def get_oss_endpoint(region: str) -> str:
        try:
            urllib.request.urlopen(f"https://oss-{region}-internal.aliyuncs.com", timeout=1)
        except HTTPError:
            pass
        except URLError:
            return f"https://oss-{region}.aliyuncs.com"
        return f"https://oss-{region}-internal.aliyuncs.com"

    def __init__(self, accesskey_id: str, accesskey_secret: str, region: str, bucket_name: str):
        super().__init__()
        self.auth = Auth(accesskey_id, accesskey_secret)
        self.bucket = Bucket(self.auth, self.get_oss_endpoint(region), bucket_name)
