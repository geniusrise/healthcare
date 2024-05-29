import csv
import logging

import networkx as nx
from tqdm import tqdm

log = logging.getLogger(__name__)


def process_stated_relationship_file(stated_relationship_file: str, G: nx.DiGraph) -> None:
    """
    Processes the SNOMED CT stated relationship file and adds the relationships to the graph.

    Args:
        stated_relationship_file (str): Path to the stated relationship file.
        G (nx.DiGraph): The NetworkX graph to which the relationships will be added.

    Returns:
        None
    """
    with open(stated_relationship_file, "rbU") as f:
        num_lines = sum(1 for _ in f)

    log.info(f"Loading stated relationships from {stated_relationship_file}")
    with open(stated_relationship_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
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
                raise ValueError(f"Error processing stated relationship {row}: {e}")
