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


def process_relationships_file(relationships_file: str, G: nx.DiGraph) -> None:
    """
    Processes the UMLS relationships file (MRREL.RRF) and adds the relationships to the graph.

    Args:
        relationships_file (str): Path to the UMLS relationships file (MRREL.RRF).
        G (nx.DiGraph): The NetworkX graph to which the relationships will be added.

    Returns:
        None
    """
    log.info(f"Loading relationships from {relationships_file}")

    with open(relationships_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in tqdm(reader):
            try:
                cui1, rel, cui2, rela, sab = row[0], row[3], row[4], row[7], row[10]
                if cui1 in G and cui2 in G:
                    G.add_edge(cui1, cui2, rel=rel, rela=rela, sab=sab)
            except Exception as e:
                log.error(f"Error processing relationship {row}: {e}")
                raise ValueError(f"Error processing relationship {row}: {e}")
