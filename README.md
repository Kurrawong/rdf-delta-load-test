# RDF Delta Load Test

For load testing Fuseki & RDF delta server

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

Create a file called `.env` in the root directory of the project
and set the desired configuration values.

> The default variables and values are shown in [.env-template](.env-template)

Run the main script

```bash
python scripts/main.py
```

1. The specified volume of RDF will be generated
1. The RDF will be submitted to RDF Delta server in the form of patch logs.
1. Fuseki will be queried the specified number of times.
1. Metrics will be printed to the console.

> The default settings will generate 1Mb of data and submit 10 queries (submitted 5 at a time)

Generated RDF will be reused across tests. If you need to regenerate it for some reason,
simply delete the `rdf` folder in the root directory. This folder is created
automatically after you run the script for the first time.

> [!IMPORTANT]
> you will need to force the regeneration of RDF if you change the indexed_property
> settings. Again, you can do this by just deleting the `rdf` folder.

