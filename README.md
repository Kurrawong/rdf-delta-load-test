# RDF Delta Load Test

For load testing RDF delta server

## Usage

load testing is run by posting data to an rdf delta endpoint and submitting sparql
queries.

The `main.py` module is the entrpoint. To run a load test with a small amount of data
from cloudshell you can do the following

> [!IMPORTANT]  
> be sure to update the URLs to point to your environment

```bash
git clone https://github.com/Kurrawong/rdf-delta-load-test.git
cd rdf-delta-load-test

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export LT__DELTA_ENDPOINT="https://rdf.delta.endpoint/"
export LT__PATCH_LOG="myds"
export LT__SPARQL_ENDPOINT="https://prez.endpoint/"
export LT__PROFILE="small_read_write"

python scripts/main.py
```

As you can see, the tests need a LT**DELTA_ENDPOINT, LT**PATCH_LOG name and LT\_\_SPARQL_ENDPOINT to
work. All of which are set in environment variables.

The LT\_\_PROFILE controls which type of test to run.

In the above example, the `small_read_write` profile generates 500 Mb of RDF data
and posts it to the RDF Delta endpoint. It also submits batches of 20 simultaneous
sparql queries for a total of 500 queries, each requesting 1000 triples.

Alternative profiles are detailed in the [profiles](##profiles) section below.

## Profiles

The following profiles are available to be set via the LT\_\_PROFILE environment
variable.

| profile            | writes                    | read                                                       |
| ------------------ | ------------------------- | ---------------------------------------------------------- |
| dev                | 1 Mb files 2 Mb total     | 10 queries, 1000 triples per query, 5 concurrent queries   |
| small_read_write   | 30 Mb files, 500 Mb total | 500 queries, 1000 triples per query, 20 concurrent queries |
| large_single_write | 100 Mb file, 100 Mb total | NA                                                         |

## Scenarios

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
