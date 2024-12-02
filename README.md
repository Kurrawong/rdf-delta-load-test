# RDF Delta Load Test

For load testing RDF delta server

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
