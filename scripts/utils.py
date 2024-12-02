import functools
import logging
import re
import time
import timeit
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx
from config import config
from rdf_graph_gen.rdf_graph_generator import generate_rdf
from rdflib import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(config.log_level)


def profile(func):
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        exec_time = timeit.default_timer()
        tracemalloc.start()
        value = func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
        logger.info(
            f"""
        {func.__name__} with args {args}:
            took {timeit.default_timer() - exec_time:.2f}s
            peak memory usage {peak / 10**6:.2f}MB
          """
        )
        tracemalloc.stop()
        return value

    return wrapper_func


def get_last_patch_id() -> str:
    response = httpx.post(
        url=config.rdf_delta_url + "$/rpc",
        json={"opid": "", "operation": "list_datasource", "arg": {}},
    )
    ds_ids = response.json().get("array", [])
    assert len(ds_ids) == 1
    ds_id = ds_ids[0]
    response = httpx.post(
        url=config.rdf_delta_url + "$/rpc",
        json={"opid": "", "operation": "describe_log", "arg": {"datasource": ds_id}},
    )
    latest = response.json().get("latest", "").lstrip("id:")
    if len(latest) == 0:
        return None
    logger.debug(f"last patch id: {latest}")
    return latest


@profile
def generate_patches(patch_size: int, total_volume: int) -> None:
    """generate random RDF data

    :param patch_size: size of indiviual patch files in MB
    :param total_volume: total volume of data to create in MB
    """

    assert (
        total_volume >= patch_size
    ), "Total volume must be greater than the patch size"
    RECORDS_PER_MB = 760
    num_patches = total_volume // patch_size
    logger.info("generating random rdf data from shapes file")
    errors = 0
    futures = []
    with ProcessPoolExecutor() as executor:
        for file in range(num_patches):
            futures.append(
                executor.submit(
                    generate_rdf,
                    shape_file=config.shape_file,
                    output_file=Path(config.rdf_folder) / f"file_{file}.ttl",
                    number=patch_size * RECORDS_PER_MB,
                )
            )
        for i, future in enumerate(futures, 1):
            try:
                logger.info(f"generating file {i}/{num_patches}")
                future.result()
            except Exception as e:
                logger.error(e)
                errors += 1
    rdf_folder_size = sum(
        int(f.stat().st_size)
        for f in Path(config.rdf_folder).glob("**/*")
        if f.is_file()
    )
    logger.info(f"generated {rdf_folder_size / (1024**2):.2f} MB of data")
    if errors > 0:
        logger.info(f"failed to generate {errors} files")
    else:
        logger.info("all files generated successfully")
    return


@profile
def submit_patches(time_between: int) -> None:
    """Submit patch logs to rdf delta server

    :param time_between: time in seconds to wait between uploads
    """
    logger.info("uploading patch logs to rdf delta server")
    errors = []
    client = httpx.Client()
    prev_id = get_last_patch_id()
    files = list(Path(config.rdf_folder).glob("*.ttl"))
    for i, file in enumerate(files, 1):
        logger.info(f"uploading file {i}/{len(files)}")
        ds = Dataset()
        ds.parse(file, format="ttl")
        patch = ds.serialize(format="patch", operation="add", header_prev=prev_id)
        logger.debug(patch)
        response = client.post(
            url=config.rdf_delta_url + config.patch_log,
            content=patch,
            headers={"content-type": "application/rdf-patch"},
        )
        logger.debug(response.text)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(e)
            errors.append(file)
        prev_id = re.findall(r"H id <(.*)>", patch)[0]
        time.sleep(time_between)

    logger.info(f"posted {len(files)} files")
    if len(errors) > 0:
        logger.info(f"failed to upload {len(errors)}/{len(files)} files")
    else:
        logger.info("all files uploaded successfully")
    return


def submit_query(query: str):
    query = Path(query)
    response = httpx.get(
        url=config.sparql_endpoint + "sparql",
        params={"query": query.read_text()},
        headers={"Content-Type": "application/sparql-query"},
    )
    response.raise_for_status()
    logger.debug(response.text)
    return


@profile
def submit_queries(query: str, n_queries: int, concurrency: int) -> None:
    futures = []
    errors = 0
    with ProcessPoolExecutor(max_workers=concurrency) as executor:
        for i in range(n_queries):
            futures.append(executor.submit(submit_query, query))

        for i, future in enumerate(futures, 1):
            try:
                logger.info(f"submitting query {i}/{n_queries}")
                future.result()
            except Exception as e:
                logger.error(e)
                errors += 1

    logger.info(f"submitted {n_queries} queries")
    if errors > 0:
        logger.info(f"failed to retrieve {errors}/{n_queries} queries")
    else:
        logger.info("all queries retrieved successfully")
    return


def services_up(required_services: list[str]) -> bool:
    try:
        for service in required_services:
            httpx.get(service).raise_for_status()
    except Exception as e:
        logger.error(e)
        return False
    return True
