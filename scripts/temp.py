import csv
import re
from pathlib import Path

with open(Path(__file__).parent / "data" / "destinations.csv") as file:
    reader = csv.reader(file)
    with open(Path(__file__).parent / "data" / "destinationsx.csv", "w") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        for row in reader:
            row[1] = re.sub(r"({|\\)", "", row[1])
            writer.writerow(row)
