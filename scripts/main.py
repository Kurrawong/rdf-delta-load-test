import logging
import sys

from config import config
from utils import (error_tracker, generate_patches, metrics, services_up,
                   submit_patches, submit_queries)

root_logger = logging.getLogger()
root_logger.addHandler(logging.StreamHandler(sys.stderr))


def main():
    delta_up, sparql_up = services_up()

    if not delta_up and not sparql_up:
        logger.error("No services are available. Exiting.")
        return

    if config.rdf_volume_mb >= 1:
        generate_patches()
        if delta_up:
            submit_patches()
    if sparql_up:
        submit_queries()

    # Only show summaries if we actually ran some tests
    if delta_up or sparql_up:
        metrics.display_summary()
        error_tracker.display_error_summary()
    else:
        logger.info("\nNo performance metrics to display - services were not available")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(config.log_level)
    logger.info(config)
    main()
