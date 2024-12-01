import logging
import os
import re
import sys
from pathlib import Path

import httpx
from rdf_graph_gen.rdf_graph_generator import generate_rdf
from rdflib import Dataset

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))
log_level = getattr(logging, os.environ.get("LOG_LEVEL", "DEBUG").upper())
logger.setLevel(log_level)


def get_last_patch_id(rdf_delta_url: str) -> str:
    response = httpx.post(
        url=rdf_delta_url + "$/rpc",
        json={"opid": "", "operation": "list_datasource", "arg": {}},
    )
    ds_ids = response.json().get("array", [])
    assert len(ds_ids) == 1
    ds_id = ds_ids[0]
    response = httpx.post(
        url=rdf_delta_url + "$/rpc",
        json={"opid": "", "operation": "describe_log", "arg": {"datasource": ds_id}},
    )
    ids = response.json().get("array", [])
    if len(ids) == 0:
        return None
    return ids[-1]


def load_data(num_files: int, num_records: int, rdf_delta_url: str, patch_log: str):
    shape_file = Path(__file__).parent.parent / "shapes/book_shape.ttl"
    rdf_folder = Path(__file__).parent.parent / "rdf"
    client = httpx.Client()
    for file in range(num_files):
        logger.debug(f"generating file {file + 1}/{num_files}")
        generate_rdf(
            shape_file=shape_file,
            output_file=rdf_folder / f"file_{file}.ttl",
            number=num_records,
        )
    prev_id = get_last_patch_id(rdf_delta_url)
    files = list(rdf_folder.glob("*.ttl"))
    for i, file in enumerate(files, 1):
        logger.debug(f"uploading file {i}/{len(files)}")
        ds = Dataset()
        ds.parse(file, format="ttl")
        patch = ds.serialize(format="patch", operation="add", header_prev=prev_id)
        logger.debug(patch)
        response = client.post(
            url=rdf_delta_url + patch_log,
            content=patch,
            headers={"content-type": "application/rdf-patch"},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(e)
        prev_id = re.findall(r"H id <(.*)>", patch)[0]


def services_up(services: list[str]) -> bool:
    for service in services:
        response = httpx.get(service)
        response.raise_for_status()
    return True


def main():
    num_files = int(os.environ.get("NUM_FILES", 2))
    num_records = os.environ.get("NUM_RECORDS", 10)
    rdf_delta_url = os.environ.get("RDF_DELTA_URL", "http://localhost:1066/")
    if not rdf_delta_url.endswith("/"):
        rdf_delta_url += "/"
    patch_log = os.environ.get("PATCH_LOG", "myds")
    assert services_up([rdf_delta_url])
    load_data(num_files, num_records, rdf_delta_url, patch_log)


if __name__ == "__main__":
    main()
