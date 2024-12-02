import logging
import sys
from pathlib import Path

from config import PROF, config
from utils import (generate_simple_patches, services_up, submit_patches,
                   submit_queries)

root_logger = logging.getLogger()
root_logger.addHandler(logging.StreamHandler(sys.stderr))


def main():

    # minimal testing config for development
    if config.profile == PROF.dev:
        assert services_up([config.delta_endpoint, config.sparql_endpoint])
        generate_simple_patches(patch_size=1, total_volume=2)
        submit_patches(time_between=0)
        submit_queries(
            query=Path(__file__).parent / "queries" / "simple_select.rq",
            n_queries=10,
            concurrency=5,
        )

    # write 500 Mb of data
    # read 20 simultaneous queries
    elif config.profile == PROF.small_read_write:
        assert services_up([config.delta_endpoint, config.sparql_endpoint])
        generate_simple_patches(patch_size=30, total_volume=500)
        submit_patches(time_between=0)
        submit_queries(
            query=Path(__file__).parent / "queries" / "simple_select.rq",
            n_queries=500,
            concurrency=20,
        )

    # write 100 mb of data, but all from one file
    # no reads
    elif config.profile == PROF.large_single_write:
        assert services_up([config.delta_endpoint])
        generate_simple_patches(patch_size=100, total_volume=100)
        submit_patches(time_between=0)
    else:
        raise NotImplementedError(f"profile {config.profile} is not implemented")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(config.log_level)
    logger.info(str(config))
    main()
