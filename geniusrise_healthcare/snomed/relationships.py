import csv
import logging

import networkx as nx
from tqdm import tqdm

log = logging.getLogger(__name__)


def process_relationship_file(relationship_file: str, G: nx.DiGraph) -> None:
    file_length = 0
    with open(relationship_file, "rbU") as f:
        num_lines = sum(1 for _ in f)

    log.info(f"Loading relationships from {relationship_file}")
    with open(relationship_file, "r") as f:  # type: ignore
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)  # type: ignore
        next(reader)  # Skip header
        for row in tqdm(reader, total=num_lines):
            try:
                source_id, dest_id, active, relationship_type, relationship_group = (
                    row[4],
                    row[5],
                    row[2],
                    row[7],
                    row[6],
                )
                if active == "1":
                    G.add_edge(
                        int(source_id),
                        int(dest_id),
                        relationship_type=relationship_type,
                        relationship_group=relationship_group,
                    )
            except Exception as e:
                raise ValueError(f"Error processing relation {row}")
