import logging
import sys

import dotenv
from config import config
from utils import (create_table, error_tracker, generate_patches, metrics,
                   services_up, submit_patches, submit_queries)

root_logger = logging.getLogger()
root_logger.addHandler(logging.StreamHandler(sys.stderr))


dotenv.load_dotenv()


def display_config_table():
    headers = ["Setting", "Value", "Description"]
    rows = [
        ["RDF Volume", config.rdf_volume_mb, "Total volume of RDF data in MB"],
        ["Query Concurrency", config.query_concurrency, "Number of concurrent queries"],
        ["Number of Queries", config.num_queries, "Total number of queries to execute"],
        ["Delta Endpoint", config.delta_endpoint, "RDF Delta server endpoint"],
        ["SPARQL Endpoint", config.sparql_endpoint, "SPARQL query endpoint"],
        ["Patch Log", config.patch_log, "RDF Delta patch log name"],
        [
            "Delay Between Patches",
            config.delay_between_patch_submissions,
            "Delay between patch submissions (seconds)",
        ],
        ["Log Level", config.log_level, "Logging verbosity level"],
        ["HTTP Timeout", config.http_timeout, "HTTP request timeout in seconds"],
        [
            "Shape File",
            config.shape_file or "None",
            "SHACL shape file for RDF generation",
        ],
    ]

    logger.info("\nConfiguration Settings\n" + create_table(headers, rows))


def main():
    delta_up, sparql_up = services_up()

    if not delta_up and not sparql_up:
        logger.error("No services are available. Exiting.")
        return

    generate_patches()
    exit()
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
    display_config_table()
    main()
