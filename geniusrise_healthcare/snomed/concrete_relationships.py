import csv
import logging

import networkx as nx
from tqdm import tqdm

log = logging.getLogger(__name__)


def process_concrete_values_file(concrete_values_file: str, G: nx.DiGraph) -> None:
    """
    Processes the SNOMED CT concrete values file and adds the relationships to the graph.

    Args:
        concrete_values_file (str): Path to the concrete values file.
        G (nx.DiGraph): The NetworkX graph to which the relationships will be added.

    Returns:
        None
    """
    with open(concrete_values_file, "rbU") as f:
        num_lines = sum(1 for _ in f)

    log.info(f"Loading concrete values from {concrete_values_file}")
    with open(concrete_values_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in tqdm(reader, total=num_lines):
            try:
                source_id, value, active, relationship_type, relationship_group = (
                    row[4],
                    row[5],
                    row[2],
                    row[7],
                    row[6],
                )
                if active == "1":
                    G.add_edge(
                        int(source_id),
                        value,
                        relationship_type=relationship_type,
                        relationship_group=relationship_group,
                    )
            except Exception as e:
                raise ValueError(f"Error processing concrete value {row}: {e}")
