# RDF Delta Load Test

For load testing RDF delta server

## Usage

load testing is run by posting data to an rdf delta endpoint and submitting sparql
queries.

The `main.py` module is the entrpoint. A small read / write test can be run like so:

```bash
export LT__DELTA_ENDPOINT="https://rdf.delta.endpoint/"
export LT__PATCH_LOG="myds"
export LT__SPARQL_ENDPOINT="https://prez.endpoint/"
export LT__PROFILE="small_read_write"

poetry run python scripts/main.py
```

As you can see, the tests need a delta endpoint, patch log name and sparql endpoint to
work. All of which are set in environment variables.

The LT\_\_PROFILE controls which type of test to run.

In the above example, the small_read_write profile generates 500 Mb of RDF data
and posts it to the RDF Delta endpoint. It also submits batches of 20 simultaneous
sparql queries for a total of 500 queries, each query requesting 1000 triples.

The other profiles are detailed in the [profiles](##profiles) section below.

## Profiles

### dev

**write**

- 1 Mb files, 2 Mb total

**read**

- 10 queries
- 1000 triples per query
- 5 concurrent queries

### small_read_write

**write**

- 30 Mb files, 500 Mb total

**read**

- 500 queries
- 1000 triples per query
- 20 concurrent queries

### large_single_write

**write**

- 100 Mb file, 100 Mb total

**read**

- NA

## scenarios

- Test with small / medium / large dataset sizes large being in the order of 1-200Gb
- Automate testing of x number of users sending requests to PrezAPI (to simulate requests from a browser)
- With or without Olis
- Test spatial queries, search queries, item page, list page

**measurements**

- time taken to load data,
- time taken to build indexes,
- time taken to execute sparql queries of different kinds,
- number of failed requests,

- cpu usage
- memory usage
- disk usage

### sub-scenarios

- rapid submission of many small patch logs
- submission of single very large patch log
