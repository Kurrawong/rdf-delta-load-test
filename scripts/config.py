import logging
import os

from dotenv import load_dotenv
from formatting import create_table

load_dotenv(override=False)

logger = logging.getLogger(__name__)


class Config:
    """load testing configuration

    :param log_level: logging level
    :param delta_endpoint: rdf delta endpoint
    :param patch_log: rdf delta data source (same as fuseki dataset name)
    :param sparql_endpoint: sparql endpoint
    :param http_timeout: timeout for http requests (in seconds)
    :param rdf_volume_mb: total volume of RDF data to generate (in MB)
    :param query_type: Type of query to submit ("simple", "fts", "geo")
    :param num_queries: how many queries to execute
    :param query_concurrency: how many queries should be executed simultaneously
    :param indexed_property_1: an rdf predicate to use in the generated RDF. It should be included in the text index for your fuseki instance
    :param indexed_property_2: as per indexed_property_1
    """

    def __init__(self):
        self.log_level: int = logging.DEBUG

        self.delta_endpoint: str = "http://localhost:1066/"
        self.patch_log: str = "myds"
        self.sparql_endpoint: str = "http://localhost:3030/myds/sparql"
        self.http_timeout: int = 60

        self.rdf_volume_mb: int = 2

        self.query_type: str = "simple"
        self.num_queries: int = 10
        self.query_concurrency: int = 5

        self.indexed_property_1: str = "https://schema.org/name"
        self.indexed_property_2: str = "https://schema.org/review"

        for k, v in vars(self).items():
            env_value = os.environ.get(f"LT__{k.upper()}", None)
            if env_value is not None:
                if k == "log_level":
                    self.__dict__[k] = getattr(logging, env_value.upper())
                elif k == "delta_endpoint":
                    self.__dict__[k] = env_value.rstrip("/") + "/"
                elif k == "sparql_endpoint":
                    self.__dict__[k] = env_value.rstrip("/")
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
