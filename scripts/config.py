import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """load testing configuration

    :param log_level: logging level
    :param delta_endpoint: rdf delta endpoint
    :param patch_log: rdf delta data source (same as fuseki dataset name)
    :param sparql_endpoint: sparql endpoint (should not end with /sparql)
    :param shape_file: optional shape file to use for rdf generation
    :param reuse_rdf: default True. to reuse previously generated data
    :param rdf_folder: where generated rdf data is stored
    :param http_timeout: timeout for http requests
    :param rdf_file_size: size of individual RDF files to generate (in MB)
    :param rdf_volume: total volume of RDF data to generate (in MB)
    :param query_template: a sparql query to use for testing the queries
    :param num_queries: how many queries to execute
    :param query_concurrency: how many queries should be executed simultaneously
    :param delay_between_patch_submissions: a delay (in seconds) between submission of RDF patch logs defaults to 0 for no delay
    """

    def __init__(self):
        self.log_level: int = logging.INFO
        self.delta_endpoint: str = "http://localhost:1066/"
        self.patch_log: str = "myds"
        self.sparql_endpoint: str = "http://localhost:8000/"
        self.shape_file: str | None = None
        self.reuse_rdf: bool = True
        self.rdf_folder: str = str(Path(__file__).parent.parent / "rdf")
        self.http_timeout: int = 60

        self.rdf_file_size: int = 1
        self.rdf_volume: int = 2

        self.query_template: str = str(
            Path(__file__).parent / "queries" / "simple_select.rq"
        )
        self.num_queries: int = 10
        self.query_concurrency: int = 5
        self.delay_between_patch_submissions: int = 0

        for k, v in vars(self).items():
            env_value = os.environ.get(f"LT__{k.upper()}", None)
            if env_value is not None:
                if k == "log_level":
                    self.__dict__[k] = getattr(logging, env_value.upper())
                elif k.endswith("endpoint"):
                    self.__dict__[k] = env_value.rstrip("/") + "/"
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
        return "CONFIG\n\t" + "\n\t".join([f"{k}: {v}" for k, v in vars(self).items()]) + "\n"


config = Config()
