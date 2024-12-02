import logging
import os
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PROF(Enum):
    dev = "dev"
    default = "default"
    small_read_write = "small_read_write"
    large_single_write = "large_single_write"


class Config:
    def __init__(self):
        self.log_level = logging.INFO
        self.profile = PROF.dev
        self.delta_endpoint = "http://localhost:1066/"
        self.patch_log = "myds"
        self.sparql_endpoint = "http://localhost:8000/"
        self.shape_file = str(Path(__file__).parent.parent / "shapes/book_shape.ttl")
        self.rdf_folder = str(Path(__file__).parent.parent / "rdf")
        self.http_timeout = 60

        for k, v in vars(self).items():
            env_value = os.environ.get(f"LT__{k.upper()}", None)
            if env_value is not None:
                if k == "log_level":
                    self.__dict__[k] = getattr(logging, env_value.upper())
                elif k == "profile":
                    self.__dict__[k] = getattr(PROF, env_value.lower())
                elif isinstance(v, bool):
                    self.__dict__[k] = self.str_to_bool(env_value)
                elif isinstance(v, int):
                    self.__dict__[k] = int(env_value)
                else:
                    self.__dict__[k] = env_value
            else:
                self.__dict__[k] = v

    def str_to_bool(self, value):
        return value.lower() in ("yes", "true", "t", "1")

    def __repr__(self):
        return "\n".join([f"{k}: {v}" for k, v in vars(self).items()])


config = Config()
