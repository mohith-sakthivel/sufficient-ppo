import yaml
from typing import Dict, Any, Union


class AttrDict(dict):
    """
    Dict-like data-structure to store experiment args
    Credits: Adapted from Danijar Hafner
    """

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def __getstate__(self) -> Dict[str, Any]:
        temp = {}
        for key, val in self.items():
            if isinstance(val, AttrDict):
                temp[key] = val.__getstate__()
            else:
                temp[key] = val
        return temp

    def __setstate__(self, state_dict: Dict[str, Any]):
        for key, val in state_dict.items():
            self[key] = val


class ProgressSchedule(yaml.YAMLObject):
    yaml_tag = 'ProgressSchedule'

    def __init__(self, value: Union[int, float]):
        self.value = value

    def __call__(self, fraction: Union[int, float]):
        return fraction * self.value

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map",
                                        {"value": data.value, "schedule": 'train-fraction'})
