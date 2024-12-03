import functools
import logging
import random
import re
import string
import time
import timeit
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import httpx
from config import config
from jinja2 import Template
from rdf_graph_gen.rdf_graph_generator import generate_rdf
from rdflib import Dataset, Graph

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
        url=config.delta_endpoint + "$/rpc",
        json={"opid": "", "operation": "list_datasource", "arg": {}},
    )
    ds_ids = response.json().get("array", [])
    assert len(ds_ids) == 1
    ds_id = ds_ids[0]
    response = httpx.post(
        url=config.delta_endpoint + "$/rpc",
        json={"opid": "", "operation": "describe_log", "arg": {"datasource": ds_id}},
    )
    latest = response.json().get("latest", "").replace("id:", "")
    if len(latest) == 0:
        return None
    logger.debug(f"last patch id: {latest}")
    return latest


def generate_simple_rdf(output_file: str, number: int) -> None:
    data = []
    for i in range(number):
        data.append(
            {
                "id": uuid4(),
                "title": "".join(
                    random.choice(string.ascii_letters) for _ in range(100)
                ),
                "description": "".join(
                    random.choice(string.ascii_letters) for _ in range(1150)
                ),
            }
        )

    template = Template(
        (Path(__file__).parent / "templates" / "rdf.ttl.j2").read_text()
    )
    g = Graph().parse(data=template.render(data=data), format="turtle")
    g.serialize(destination=output_file, format="turtle")
    return


@profile
def generate_patches() -> None:
    """generate random RDF data optionally using a  SHACL shape file

    WARNING: Shape file generation is very slow

    outputs are stored under the config.rdf_folder directory in a
    subfolder called {rdf_file_size}-{rdf_volume}. In this way, the
    generated data can be reused.
    """

    if not Path(config.rdf_folder).exists():
        Path(config.rdf_folder).mkdir()
    out_folder = Path(config.rdf_folder) / f"{config.rdf_file_size}-{config.rdf_volume}"
    if config.reuse_rdf and out_folder.exists():
        logger.info(
            "reusing previously generated data, to force regeneration you can `export LT__REUSE_RDF=false`"
        )
        return
    if not out_folder.exists():
        out_folder.mkdir()
    assert (
        config.rdf_volume >= config.rdf_file_size
    ), "RDF volume must be greater than the patch size"

    RECORDS_PER_MB = 760
    num_patches = config.rdf_volume // config.rdf_file_size
    logger.info("generating random rdf data from shapes file")
    errors = 0
    futures = []
    with ProcessPoolExecutor() as executor:
        for file in range(num_patches):
            if config.shape_file:
                futures.append(
                    executor.submit(
                        generate_rdf,
                        shape_file=config.shape_file,
                        output_file=out_folder / f"file_{file}.ttl",
                        number=config.rdf_file_size * RECORDS_PER_MB,
                    )
                )
            else:
                futures.append(
                    executor.submit(
                        generate_simple_rdf,
                        output_file=out_folder / f"file_{file}.ttl",
                        number=config.rdf_file_size * RECORDS_PER_MB,
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
def submit_patches() -> None:
    logger.info("uploading patch logs to rdf delta server")
    errors = []
    client = httpx.Client(timeout=config.http_timeout)
    prev_id = get_last_patch_id()
    files = list((Path(config.rdf_folder) / f"{config.rdf_file_size}-{config.rdf_volume}").glob("*.ttl"))
    for i, file in enumerate(files, 1):
        logger.info(f"uploading file {i}/{len(files)}")
        ds = Dataset()
        ds.parse(file, format="ttl")
        patch = ds.serialize(format="patch", operation="add", header_prev=prev_id)
        logger.debug(patch)
        response = client.post(
            url=config.delta_endpoint + config.patch_log,
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
        time.sleep(config.delay_between_patch_submissions)

    logger.info(f"posted {len(files)} files")
    if len(errors) > 0:
        logger.info(f"failed to upload {len(errors)}/{len(files)} files")
    else:
        logger.info("all files uploaded successfully")
    return


def submit_query(client: httpx.Client):
    query = Path(config.query_template)
    response = client.get(
        url=config.sparql_endpoint + "sparql",
        params={"query": query.read_text()},
        headers={"Content-Type": "application/sparql-query"},
    )
    response.raise_for_status()
    logger.debug(response.text)
    return


@profile
def submit_queries() -> None:
    client = httpx.Client(timeout=config.http_timeout)
    futures = []
    errors = 0
    with ThreadPoolExecutor(max_workers=config.query_concurrency) as executor:
        for i in range(config.num_queries):
            futures.append(executor.submit(submit_query, client=client))

        for i, future in enumerate(futures, 1):
            try:
                logger.info(f"submitting query {i}/{config.num_queries}")
                future.result()
            except Exception as e:
                logger.error(e)
                errors += 1

    logger.info(f"submitted {config.num_queries} queries")
    if errors > 0:
        logger.info(f"failed to retrieve {errors}/{config.num_queries} queries")
    else:
        logger.info("all queries retrieved successfully")
    return


def services_up() -> tuple[bool, bool]:
    """Check if RDF delta and SPARQL endpoints are up

    :return: a tuple indicating if rdf delta and sparql endpoints are up (in that order)
    """
    delta_up = True
    sparql_up = True
    try:
        response = httpx.get(config.delta_endpoint)
        logger.debug(f"delta status: {response.status_code}")
        response.raise_for_status()
    except Exception as e:
        logger.error(e)
        logger.error("delta endpoint is not available")
        delta_up = False

    try:
        response = httpx.get(config.sparql_endpoint)
        logger.debug(f"sparql status: {response.status_code}")
        response.raise_for_status()
    except Exception as e:
        logger.error(e)
        logger.error("sparql endpoint is not available")
        sparql_up = False

    return delta_up, sparql_up
