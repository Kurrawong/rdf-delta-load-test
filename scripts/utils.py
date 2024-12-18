from pathlib import Path
from uuid import uuid4
import functools
import logging
import random
import re
import string
import time
import timeit
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import httpx
from config import config
from jinja2 import Template
from rdf_graph_gen.rdf_graph_generator import generate_rdf
from rdflib import Dataset, Graph

logger = logging.getLogger(__name__)
logger.setLevel(config.log_level)


class Metrics:
    def __init__(self):
        self.start_time = time.time()
        self.generation_metrics = {}
        self.upload_metrics = {}
        self.query_metrics = {}
        self.container_config = {}
    
    def add_container_config(self, container_stats):
        """Add initial container configuration"""
        if container_stats:
            self.container_config = {
                "Delta Server Memory Limit (MB)": f"{container_stats['docker-rdf-delta-server-1']['memory_limit_mb']:.2f}",
                "Delta Server Initial Memory (MB)": f"{container_stats['docker-rdf-delta-server-1']['memory_usage_mb']:.2f}",
                "Delta Server Initial CPU (%)": f"{container_stats['docker-rdf-delta-server-1']['cpu_percent']:.1f}",
                "Fuseki Server Memory Limit (MB)": f"{container_stats['docker-rdf-delta-fuseki-server-1']['memory_limit_mb']:.2f}",
                "Fuseki Server Initial Memory (MB)": f"{container_stats['docker-rdf-delta-fuseki-server-1']['memory_usage_mb']:.2f}",
                "Fuseki Server Initial CPU (%)": f"{container_stats['docker-rdf-delta-fuseki-server-1']['cpu_percent']:.1f}",
                "Prez Memory Limit (MB)": f"{container_stats['docker-prez-1']['memory_limit_mb']:.2f}",
                "Prez Initial Memory (MB)": f"{container_stats['docker-prez-1']['memory_usage_mb']:.2f}",
                "Prez Initial CPU (%)": f"{container_stats['docker-prez-1']['cpu_percent']:.1f}"
            }
    
    def add_generation_metrics(self, total_files, successful, failed, total_size, container_stats=None):
        total_time = time.time() - self.start_time
        self.generation_metrics = {
            "Total Files Generated": total_files,
            "Successful Generations": successful,
            "Failed Generations": failed,
            "Generation Size (MB)": f"{total_size / 1024 / 1024:.2f}",
            "Generation Time (s)": f"{total_time:.2f}",
            "Generation Rate (MB/s)": f"{(total_size / 1024 / 1024) / total_time:.2f}"
        }
        if container_stats:
            self.generation_metrics.update({
                "Delta Server Memory (MB)": f"{container_stats['docker-rdf-delta-server-1']['memory_usage_mb']:.2f}",
                "Delta Server CPU (%)": f"{container_stats['docker-rdf-delta-server-1']['cpu_percent']:.1f}",
                "Fuseki Server Memory (MB)": f"{container_stats['docker-rdf-delta-fuseki-server-1']['memory_usage_mb']:.2f}",
                "Fuseki Server CPU (%)": f"{container_stats['docker-rdf-delta-fuseki-server-1']['cpu_percent']:.1f}"
            })
    
    def add_upload_metrics(self, total_files, successful, failed, total_size, exec_time=None, peak_memory=None, container_stats=None):
        total_time = time.time() - self.start_time
        self.upload_metrics = {
            "Total Files Uploaded": total_files,
            "Successful Uploads": successful,
            "Failed Uploads": failed,
            "Upload Size (MB)": f"{total_size / 1024 / 1024:.2f}",
            "Upload Time (s)": f"{total_time:.2f}",
            "Upload Rate (MB/s)": f"{(total_size / 1024 / 1024) / total_time:.2f}",
            "Execution Time (s)": f"{exec_time:.2f}" if exec_time else "N/A",
            "Peak Memory (MB)": f"{peak_memory:.2f}" if peak_memory else "N/A"
        }
        if container_stats:
            self.upload_metrics.update({
                "Delta Server Memory (MB)": f"{container_stats['docker-rdf-delta-server-1']['memory_usage_mb']:.2f}",
                "Delta Server CPU (%)": f"{container_stats['docker-rdf-delta-server-1']['cpu_percent']:.1f}",
                "Fuseki Server Memory (MB)": f"{container_stats['docker-rdf-delta-fuseki-server-1']['memory_usage_mb']:.2f}",
                "Fuseki Server CPU (%)": f"{container_stats['docker-rdf-delta-fuseki-server-1']['cpu_percent']:.1f}"
            })
    
    def add_query_metrics(self, total_queries, successful, failed, total_time, avg_time, qps, exec_time=None, peak_memory=None, container_stats=None):
        self.query_metrics = {
            "Total Queries": total_queries,
            "Successful Queries": successful,
            "Failed Queries": failed,
            "Total Query Time (s)": f"{total_time:.2f}",
            "Avg Query Time (s)": f"{avg_time:.2f}",
            "Queries/Second": f"{qps:.2f}",
            "Execution Time (s)": f"{exec_time:.2f}" if exec_time else "N/A",
            "Peak Memory (MB)": f"{peak_memory:.2f}" if peak_memory else "N/A"
        }
        if container_stats:
            self.query_metrics.update({
                "Delta Server Memory (MB)": f"{container_stats['docker-rdf-delta-server-1']['memory_usage_mb']:.2f}",
                "Delta Server CPU (%)": f"{container_stats['docker-rdf-delta-server-1']['cpu_percent']:.1f}",
                "Fuseki Server Memory (MB)": f"{container_stats['docker-rdf-delta-fuseki-server-1']['memory_usage_mb']:.2f}",
                "Fuseki Server CPU (%)": f"{container_stats['docker-rdf-delta-fuseki-server-1']['cpu_percent']:.1f}",
                "Prez Memory (MB)": f"{container_stats['docker-prez-1']['memory_usage_mb']:.2f}",
                "Prez CPU (%)": f"{container_stats['docker-prez-1']['cpu_percent']:.1f}"
            })
    
    def display_summary(self):
        alignments = ['left', 'right']
        
        # Container Configuration Table
        if self.container_config:
            headers = ["Container Configuration", "Value"]
            rows = list(self.container_config.items())
            logger.info("\n" + create_table(headers, rows, alignments))
        
        # Upload Metrics Table
        if self.upload_metrics:
            headers = ["Upload Metrics", "Value"]
            rows = list(self.upload_metrics.items())
            logger.info("\n" + create_table(headers, rows, alignments))
        
        # Query Metrics Table
        if self.query_metrics:
            headers = ["Query Metrics", "Value"]
            rows = list(self.query_metrics.items())
            logger.info("\n" + create_table(headers, rows, alignments))


class ErrorTracker:
    def __init__(self):
        self.query_errors = {
            'connection': 0,    # Connection/network errors
            'timeout': 0,       # Request timeouts
            'http_400': 0,      # Bad requests (malformed queries)
            'http_500': 0,      # Server errors
            'other': 0          # Uncategorized errors
        }
        self.upload_errors = {
            'connection': 0,    # Connection/network errors
            'parse': 0,         # TTL parsing errors
            'patch': 0,         # Patch creation/serialization errors
            'http_400': 0,      # Bad requests
            'http_500': 0,      # Server errors
            'other': 0          # Uncategorized errors
        }
    
    def add_query_error(self, error: Exception):
        if isinstance(error, httpx.ConnectError):
            self.query_errors['connection'] += 1
        elif isinstance(error, httpx.TimeoutException):
            self.query_errors['timeout'] += 1
        elif isinstance(error, httpx.HTTPStatusError):
            if error.response.status_code == 400:
                self.query_errors['http_400'] += 1
            elif error.response.status_code == 500:
                self.query_errors['http_500'] += 1
            else:
                self.query_errors['other'] += 1
        else:
            self.query_errors['other'] += 1
    
    def add_upload_error(self, error: Exception):
        if isinstance(error, httpx.ConnectError):
            self.upload_errors['connection'] += 1
        elif isinstance(error, (ValueError, SyntaxError)):
            self.upload_errors['parse'] += 1
        elif isinstance(error, httpx.HTTPStatusError):
            if error.response.status_code == 400:
                self.upload_errors['http_400'] += 1
            elif error.response.status_code == 500:
                self.upload_errors['http_500'] += 1
            else:
                self.upload_errors['other'] += 1
        else:
            self.upload_errors['other'] += 1
    
    def display_error_summary(self):
        if not any(self.query_errors.values()) and not any(self.upload_errors.values()):
            return
        
        headers = ["Error Type", "Count"]
        rows = []
        
        if any(self.upload_errors.values()):
            rows.extend([
                ["Upload Errors", ""],
                *[(f"  {k.replace('_', ' ').title()}", v) 
                  for k, v in self.upload_errors.items() if v > 0]
            ])
        
        if any(self.query_errors.values()):
            if rows:  # Add spacing if we have upload errors
                rows.append(["", ""])
            rows.extend([
                ["Query Errors", ""],
                *[(f"  {k.replace('_', ' ').title()}", v) 
                  for k, v in self.query_errors.items() if v > 0]
            ])
        
        if rows:
            logger.info("\nError Summary\n" + create_table(headers, rows))

# Create global instances
metrics = Metrics()
error_tracker = ErrorTracker()


def profile(func):
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        container_stats = get_container_stats([
            'docker-rdf-delta-server-1',
            'docker-rdf-delta-fuseki-server-1',
            'docker-prez-1'
        ])
        
        # Add container configuration if not already added
        if not metrics.container_config and container_stats:
            metrics.add_container_config(container_stats)
        
        exec_time = timeit.default_timer()
        tracemalloc.start()
        value = func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / 10**6
        exec_time = timeit.default_timer() - exec_time
        
        # Log basic stats
        log_msg = f"""
        {func.__name__} with args {args}:
            took {exec_time:.2f}s
            peak memory usage {peak_mb:.2f}MB
        """
        logger.info(log_msg)
        tracemalloc.stop()

        # Update metrics with container stats
        if func.__name__ == 'submit_queries' and isinstance(value, dict):
            metrics.add_query_metrics(
                total_queries=value['total_queries'],
                successful=value['successful'],
                failed=value['failed'],
                total_time=value['total_time'],
                avg_time=value['avg_time'],
                qps=value['qps'],
                exec_time=exec_time,
                peak_memory=peak_mb,
                container_stats=container_stats
            )
        elif func.__name__ == 'submit_patches' and isinstance(value, dict):
            metrics.add_upload_metrics(
                total_files=value['total_files'],
                successful=value['successful'],
                failed=value['failed'],
                total_size=value['total_size'],
                exec_time=exec_time,
                peak_memory=peak_mb,
                container_stats=container_stats
            )

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
    
    # Calculate metrics
    total_time = time.time() - start_time
    avg_time_per_file = total_time / num_patches
    throughput = rdf_folder_size / total_time / 1024 / 1024  # MB/s
    
    # Create and display the results table
    headers = ["Metric", "Value"]
    rows = [
        ["Total Files", num_patches],
        ["Successful Generations", num_patches - errors],
        ["Failed Generations", errors],
        ["Total Size (MB)", f"{rdf_folder_size / 1024 / 1024:.2f}"],
        ["Total Time (s)", f"{total_time:.2f}"],
        ["Avg Time/File (s)", f"{avg_time_per_file:.2f}"],
        ["Throughput (MB/s)", f"{throughput:.2f}"]
    ]
    
    logger.info("\nRDF Generation Results\n" + create_table(headers, rows))
    
    # Instead of displaying table, add to metrics
    metrics.add_generation_metrics(
        total_files=num_patches,
        successful=num_patches - errors,
        failed=errors,
        total_size=rdf_folder_size,
        container_stats=container_stats
    )
    return


def create_table(headers: list, rows: list, alignments: list = None) -> str:
    """Create a text-based table with headers and rows"""
    if alignments is None:
        alignments = ['left'] * len(headers)
    
    def split_number(value):
        try:
            value = str(value).replace(',', '')
            if '.' in value:
                left, right = value.split('.')
                return left, right
            return value, ''
        except:
            return value, ''

    # Find maximum widths for decimal alignment
    max_left_width = 0
    max_right_width = 0
    for row in rows:
        if row[1]:  # Skip empty values
            left, right = split_number(str(row[1]))
            max_left_width = max(max_left_width, len(left))
            max_right_width = max(max_right_width, len(right))

    # Calculate column widths
    column_widths = [35, max_left_width + (max_right_width > 0 and max_right_width + 1 or 0) + 2]
    
    def format_cell(content, width, alignment, col_idx):
        content = str(content)
        if alignment == 'right' and col_idx == 1 and content:  # Number column
            return content.rjust(width)
        elif alignment == 'center':
            return content.center(width)
        return content.ljust(width)

    separator = "+" + "+".join(["-" * width for width in column_widths]) + "+"
    header_row = "|" + "|".join(format_cell(h, w, 'center', i) 
                               for i, (h, w) in enumerate(zip(headers, column_widths))) + "|"
    
    formatted_rows = []
    for row in rows:
        formatted_row = "|" + "|".join(format_cell(cell, width, align, i) 
                                     for i, (cell, width, align) in enumerate(zip(row, column_widths, alignments))) + "|"
        formatted_rows.append(formatted_row)
    
    return "\n".join([separator, header_row, separator, *formatted_rows, separator])


@profile
def submit_patches() -> None:
    logger.info("uploading patch logs to rdf delta server")
    errors = []
    start_time = time.time()
    total_size = 0
    client = httpx.Client(timeout=config.http_timeout)
    prev_id = get_last_patch_id()
    files = list((Path(config.rdf_folder) / f"{config.rdf_file_size}-{config.rdf_volume}").glob("*.ttl"))
    
    successful_uploads = 0
    failed_uploads = 0
    
    for i, file in enumerate(files, 1):
        file_size = file.stat().st_size
        total_size += file_size
        logger.info(f"uploading file {i}/{len(files)}")
        ds = Dataset()
        ds.parse(file, format="ttl")
        patch = ds.serialize(format="patch", operation="add", header_prev=prev_id)
        patch = patch.split("\n")
        newlines = []
        for line in patch:
            if "H prev" in line:
                line += " ."
            newlines.append(line)
        #patch = "\n".join(patch)
        patch = "\n".join(newlines)
        # logger.debug(patch)
        response = client.post(
            url=config.delta_endpoint + config.patch_log,
            content=patch,
            headers={"content-type": "application/rdf-patch"},
        )
        logger.debug(response.text)
        try:
            response.raise_for_status()
            successful_uploads += 1
        except Exception as e:
            error_tracker.add_upload_error(e)
            failed_uploads += 1
            errors.append(file)
        prev_id = re.findall(r"H id <(.*)>", patch)[0]
        time.sleep(config.delay_between_patch_submissions)
    
    # Calculate metrics
    total_time = time.time() - start_time
    
    # Return metrics dict that will be updated by the profile decorator
    metrics_dict = {
        'total_files': len(files),
        'successful': successful_uploads,
        'failed': failed_uploads,
        'total_size': total_size
    }
    return metrics_dict


def submit_query(client: httpx.Client):
    """Execute a single query and return its execution time"""
    start_time = time.time()
    query = Path(config.query_template)
    response = client.get(
        url=config.sparql_endpoint,
        params={"query": query.read_text()},
        headers={"Content-Type": "application/sparql-query"},
    )
    response.raise_for_status()
    query_time = time.time() - start_time
    logger.debug(f"Query took {query_time:.2f}s")
    return query_time


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with 5-space integer part and fixed decimal places"""
    # Split into integer and decimal parts
    int_part = int(value)
    decimal_part = abs(value - int_part)
    
    # Format integer part to take up exactly 5 spaces (right-aligned)
    formatted_int = f"{int_part:5d}"
    
    # Format decimal part with specified precision
    formatted_decimal = f"{decimal_part:.{decimals}f}"[2:]  # Remove "0."
    
    return f"{formatted_int}.{formatted_decimal}"


@profile
def submit_queries() -> None:
    client = httpx.Client(timeout=config.http_timeout)
    futures = []
    errors = 0
    successful_queries = 0
    query_times = []  # Track individual query times
    
    with ThreadPoolExecutor(max_workers=config.query_concurrency) as executor:
        # Submit all queries
        for i in range(config.num_queries):
            futures.append(executor.submit(submit_query, client=client))

        # Collect results
        for i, future in enumerate(futures, 1):
            try:
                logger.info(f"submitting query {i}/{config.num_queries}")
                query_time = future.result()  # Get the actual query time
                query_times.append(query_time)
                successful_queries += 1
            except Exception as e:
                error_tracker.add_query_error(e)
                errors += 1

    # Calculate execution-specific metrics
    total_query_time = sum(query_times)  # Total time spent in actual queries
    avg_query_time = total_query_time / len(query_times) if query_times else 0
    qps = len(query_times) / total_query_time if total_query_time > 0 else 0
    
    # Return metrics dict
    metrics_dict = {
        'total_queries': config.num_queries,
        'successful': successful_queries,
        'failed': errors,
        'total_time': total_query_time,
        'avg_time': avg_query_time,
        'qps': qps
    }
    return metrics_dict


def services_up() -> tuple[bool, bool]:
    """Check if RDF delta and SPARQL endpoints are up"""
    delta_up = True
    sparql_up = True
    
    logger.info("Checking service availability...")
    
    try:
        response = httpx.get(config.delta_endpoint)
        logger.debug(f"Delta server status: {response.status_code}")
        response.raise_for_status()
        logger.info("✓ RDF Delta server is available")
    except Exception as e:
        logger.error(f"✗ RDF Delta server is not available: {str(e)}")
        delta_up = False

    try:
        response = httpx.get(config.sparql_endpoint + "?query=ask%20{}")
        logger.debug(f"SPARQL endpoint status: {response.status_code}")
        response.raise_for_status()
        logger.info("✓ SPARQL endpoint is available")
    except Exception as e:
        logger.error(f"✗ SPARQL endpoint is not available: {str(e)}")
        sparql_up = False

    if not delta_up and not sparql_up:
        logger.error("No required services are available")
    
    return delta_up, sparql_up


def get_container_stats(container_names: list[str]) -> dict:
    """Get memory and CPU stats for specified containers if running in Docker"""
    try:
        import docker
        client = docker.from_env()
        logger.info("Docker client created successfully")
    except (ImportError, ModuleNotFoundError):
        logger.info("Docker package not installed - skipping container stats")
        return {}
    except Exception as e:
        logger.info(f"Docker not available - {str(e)}")
        return {}
    
    stats = {}
    for name in container_names:
        try:
            container = client.containers.get(name)
            logger.info(f"Found container: {name}")
            if container.status == 'running':
                raw_stats = container.stats(stream=False)
                # Calculate memory usage
                mem_stats = raw_stats['memory_stats']
                mem_usage = mem_stats.get('usage', 0)
                mem_limit = mem_stats.get('limit', 1)
                mem_percent = (mem_usage / mem_limit) * 100
                
                # Calculate CPU usage
                cpu_stats = raw_stats['cpu_stats']
                prev_cpu = raw_stats['precpu_stats']
                cpu_delta = cpu_stats['cpu_usage']['total_usage'] - prev_cpu['cpu_usage']['total_usage']
                system_delta = cpu_stats['system_cpu_usage'] - prev_cpu['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta) * 100 * cpu_stats['online_cpus']
                
                stats[name] = {
                    'memory_usage_mb': mem_usage / (1024 * 1024),
                    'memory_limit_mb': mem_limit / (1024 * 1024),
                    'memory_percent': mem_percent,
                    'cpu_percent': cpu_percent
                }
                logger.info(f"Got stats for {name}: {stats[name]}")
        except Exception as e:
            logger.info(f"Skipping Docker stats for {name}: {e}")
            
    return stats
