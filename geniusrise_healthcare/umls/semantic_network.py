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


def process_semantic_network_files(srdef_file: str, srstr_file: str, G: nx.DiGraph) -> None:
    """
    Processes the UMLS Semantic Network files (SRDEF and SRSTR) and adds the semantic types and relationships to the graph.

    Args:
        srdef_file (str): Path to the UMLS semantic type definitions file (SRDEF).
        srstr_file (str): Path to the UMLS semantic relationship file (SRSTR).
        G (nx.DiGraph): The NetworkX graph to which the semantic network will be added.

    Returns:
        None
    """
    log.info(f"Loading semantic type definitions from {srdef_file}")
    with open(srdef_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in tqdm(reader):
            try:
                tui, name, description = row[0], row[1], row[2]
                G.add_node(tui, name=name, description=description)
            except Exception as e:
                log.error(f"Error processing semantic type definition {row}: {e}")
                raise ValueError(f"Error processing semantic type definition {row}: {e}")

    log.info(f"Loading semantic relationships from {srstr_file}")
    with open(srstr_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in tqdm(reader):
            try:
                tui1, rel, tui2 = row[0], row[1], row[2]
                if tui1 in G and tui2 in G:
                    G.add_edge(tui1, tui2, rel=rel)
            except Exception as e:
                log.error(f"Error processing semantic relationship {row}: {e}")
                raise ValueError(f"Error processing semantic relationship {row}: {e}")
