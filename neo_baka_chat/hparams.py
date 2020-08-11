from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class BaseHyperParams:
    def dumps(self) -> dict:
        return asdict(self)

    # noinspection PyArgumentList
    @classmethod
    def loads(cls, obj: dict) -> "BaseHyperParams":
        _fields = fields(cls)
        keys = [field.name for field in _fields]
        allowed_kwargs = {k: v for k, v in obj.items() if k in keys}
        return cls(**allowed_kwargs)
