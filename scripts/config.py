import logging
from formatting import create_table
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

logger = logging.getLogger(__name__)


class Config:
    """load testing configuration

    :param log_level: logging level
    :param delta_endpoint: rdf delta endpoint
    :param patch_log: rdf delta data source (same as fuseki dataset name)
    :param sparql_endpoint: sparql endpoint
    :param rdf_dir: where generated rdf data is stored (relative to the project root)
    :param http_timeout: timeout for http requests (in seconds)
    :param rdf_volume_mb: total volume of RDF data to generate (in MB)
    :param num_queries: how many queries to execute
    :param query_concurrency: how many queries should be executed simultaneously
    :param delay_between_patch_submissions: a delay (in seconds) between submission of RDF patch logs defaults to 0 for no delay
    """

    def __init__(self):
        self.log_level: int = logging.INFO

        self.delta_endpoint: str = "http://localhost:1066/"
        self.patch_log: str = "myds"
        self.sparql_endpoint: str = "http://localhost:8000/sparql"
        self.http_timeout: int = 60

        self.rdf_dir: str = str(Path(__file__).parent.parent / "rdf")
        self.rdf_volume_mb: int = 2

        self.query_dir: str = str(Path(__file__).parent / "queries")
        self.num_queries: int = 10
        self.query_concurrency: int = 5
        self.delay_between_patch_submissions: int = 0

        for k, v in vars(self).items():
            env_value = os.environ.get(f"LT__{k.upper()}", None)
            if env_value is not None:
                if k == "log_level":
                    self.__dict__[k] = getattr(logging, env_value.upper())
                elif k == "delta_endpoint":
                    self.__dict__[k] = env_value.rstrip("/") + "/"
                elif k == "sparql_endpoint":
                    self.__dict__[k] = env_value.rstrip("/")
                elif k == "rdf_dir":
                    self.__dict__[k] = str(Path(__file__).parent.parent / env_value)
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
        docstring = self.__doc__
        doc_lines = docstring.splitlines()
        rows = []
        for line in doc_lines:
            if ":param" in line:
                param_name, param_info = line.split(":", 2)[1:]
                param_name = param_name.split("param ", 1)[1].strip()
                param_info = param_info.strip()
                param_value = eval(f"self.{param_name}")
                rows.append([param_name, param_value, param_info])
        headers = ["Setting", "Value", "Description"]
        return create_table(headers, rows)


config = Config()
