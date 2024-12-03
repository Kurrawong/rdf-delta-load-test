import logging
import sys

from config import config
from utils import generate_patches, services_up, submit_patches, submit_queries

root_logger = logging.getLogger()
root_logger.addHandler(logging.StreamHandler(sys.stderr))


def main():
    delta_up, sparql_up = services_up()
    generate_patches()
    if delta_up:
        submit_patches()
    if sparql_up:
        submit_queries()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(config.log_level)
    logger.info(str(config))
    main()
