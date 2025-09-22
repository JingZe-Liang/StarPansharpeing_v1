from dataclasses import dataclass, field, fields, replace
from typing import TypeVar

DataclassT = TypeVar("DataclassT")


def filter_out_for_dataclass(dataclass_type: type[DataclassT], data: dict[str, object]):
    datacls_keys = {f.name for f in fields(dataclass_type)}
    datacls_args = {}
    for k in list(data.keys()):
        if k not in datacls_keys:
            datacls_args[k] = data.pop(k)
    return datacls_args


def replace_in_dataclass(instance: DataclassT, **changes: object):
    datacls_args = filter_out_for_dataclass(type(instance), changes)
    return replace(instance, **datacls_args)
