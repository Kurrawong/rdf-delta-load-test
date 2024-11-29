from pathlib import Path

from rdf_graph_gen.rdf_graph_generator import generate_rdf

num_records = 10
shape_file = Path(__file__).parent.parent / "shapes/book_shape.ttl"
out_file = Path(__file__).parent.parent / "rdf/books.ttl"

generate_rdf(
    shape_file=shape_file,
    output_file=out_file,
    number=num_records,
)
