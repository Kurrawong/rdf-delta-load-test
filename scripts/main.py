import logging
import sys
from pathlib import Path

from config import PROF, config
from utils import (generate_simple_patches, services_up, submit_patches,
                   submit_queries)

root_logger = logging.getLogger()
root_logger.addHandler(logging.StreamHandler(sys.stderr))


def main():
    if config.profile == PROF.dev:
        assert services_up(
            [config.rdf_delta_url, config.sparql_endpoint]
        ), "ERROR: Please check that all services are online"
        generate_simple_patches(patch_size=1, total_volume=2)
        submit_patches(time_between=0)
        submit_queries(
            query=Path(__file__).parent / "queries" / "simple_select.rq",
            n_queries=10,
            concurrency=5,
        )
    else:
        raise NotImplementedError(f"profile {config.profile} is not implemented")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(config.log_level)
    logger.info(str(config))
    main()
