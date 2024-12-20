# RDF Delta Load Test

For load testing RDF delta server

## Installation

```bash
# clone the repo
git clone https://github.com/Kurrawong/rdf-delta-load-test.git
cd rdf-delta-load-test

# install python dependencies into a virtual environment
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

```

## Usage

### Pre baked configurations

The load test configurations are managed using [go-task](https://taskfile.dev). this
makes it easier to set and override environment variables for each scenario.

> [!NOTE]  
> if you can't or dont' want to install go-task then you can still run the load test
> you just need to specify the configuration for each scenario manually. See the
> [manual configuration](#manual-configuration) section below.

```bash
# install go-task
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
```

You can see a list of all the pre-baked configurations by running:

```bash
task --list-all
```

To run the small read write configuration you would run:

```bash
task small_rw
```

### Manual configurations

You can set up any configuration you want by first setting the appropriate environment
variables and then running the `scripts/main.py` script.

For example, to run the load test against a remote rdf delta server with the same
configuration as the `small_rw` task you could run:

```bash
export LT__DELTA_ENDPOINT="http://my.delta.endpoint/"
export LT__PATCH_LOG="myds"
export LT__SPARQL_ENDPOINT="http://my.sparql.endpoint/"

export LT__RDF_FILE_SIZE=30
export LT__RDF_VOLUME=500

export LT__NUM_QUERIES=500
export LT__QUERY_CONCURRENCY=20

python scripts/main.py
```

And to do the same in PowerShell

```powershell
$env:LT__DELTA_ENDPOINT = "http://my.delta.endpoint/"
$env:LT__PATCH_LOG = "myds"
$env:LT__SPARQL_ENDPOINT = "http://my.sparql.endpoint/"

$env:LT__RDF_FILE_SIZE = 30
$env:LT__RDF_VOLUME = 500

$env:LT__NUM_QUERIES = 500
$env:LT__QUERY_CONCURRENCY = 20

python scripts/main.py
```

## Configuration

The pre baked configurations are stored in [Taskfile.yaml](./Taskfile.yaml) and can be
viewed by looking in that file.

The list of run configurations can be extended by editing the [Taskfile.yaml](./Taskfile.yaml)

The full list of configuration options and their default values is given below

| Variable                              | Description                                                              | Default                    |
| ------------------------------------- | ------------------------------------------------------------------------ | -------------------------- |
| `LT__LOG_LEVEL`                       | logging level                                                            | `info`                     |
| `LT__DELTA_ENDPOINT`                  | rdf delta endpoint                                                       | `"http://localhost:1066/"` |
| `LT__PATCH_LOG`                       | rdf delta data source (same as fuseki dataset name)                      | `"myds"`                   |
| `LT__SPARQL_ENDPOINT`                 | sparql endpoint (should not end with /sparql)                            | `"http://localhost:8000/"` |
| `LT__SHAPE_FILE`                      | optional shape file to use for rdf generation                            | `None`                     |
| `LT__REUSE_RDF`                       | default True. to reuse previously generated data                         | `True`                     |
| `LT__RDF_FOLDER`                      | where generated rdf data is stored                                       | `rdf`                      |
| `LT__HTTP_TIMEOUT`                    | timeout for http requests                                                | `60`                       |
| `LT__RDF_FILE_SIZE`                   | size of individual RDF files to generate (in MB)                         | `1`                        |
| `LT__RDF_VOLUME`                      | total volume of RDF data to generate (in MB)                             | `2`                        |
| `LT__QUERY_TEMPLATE`                  | a sparql query to use for testing the queries                            | `queries/simple_select.rq` |
| `LT__NUM_QUERIES`                     | how many queries to execute                                              | `10`                       |
| `LT__QUERY_CONCURRENCY`               | how many queries should be executed simultaneously                       | `5`                        |
| `LT__DELAY_BETWEEN_PATCH_SUBMISSIONS` | a delay (in seconds) between submission of RDF patch logs, defaults to 0 | `0`                        |

## Scenarios

- Test with small / medium / large dataset sizes large being in the order of 1-200Gb
- Automate testing of x number of users sending requests to PrezAPI (to simulate requests from a browser)
- With or without Olis
- Test spatial queries, search queries, item page, list page

**measurements**

- time taken to load data,
- time taken to execute sparql queries of different kinds,
- number of failed requests,

- cpu usage
- memory usage
- disk usage

### sub-scenarios

- rapid submission of many small patch logs
- submission of single very large patch log
