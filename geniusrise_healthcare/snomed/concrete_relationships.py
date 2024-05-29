# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
    with open(concrete_values_file, "r") as f:  # type: ignore
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)  # type: ignore
        next(reader)
        for row in tqdm(reader, total=num_lines):
            try:
                source_id, value, active, relationship_type, relationship_group, characteristic_type, refinability = (
                    row[4],
                    row[5],
                    row[2],
                    row[7],
                    row[6],
                    row[8],
                    row[9],
                )
                if active == "1":
                    G.add_edge(
                        int(source_id),
                        value,
                        relationship_type=relationship_type,
                        relationship_group=relationship_group,
                        characteristic_type=characteristic_type,
                        refinability=refinability,
                    )
            except Exception as e:
                log.error(f"Error processing concrete value {row}: {e}")
                raise ValueError(f"Error processing concrete value {row}: {e}")
