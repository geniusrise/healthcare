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
from typing import Dict

import networkx as nx
from tqdm import tqdm

log = logging.getLogger(__name__)


def process_concepts_file(concepts_file: str, G: nx.DiGraph, cui_to_concept: Dict[str, Dict]) -> None:
    """
    Processes the UMLS concepts file (MRCONSO.RRF) and adds the concepts to the graph.

    Args:
        concepts_file (str): Path to the UMLS concepts file (MRCONSO.RRF).
        G (nx.DiGraph): The NetworkX graph to which the concepts will be added.
        cui_to_concept (Dict[str, Dict]): Dictionary mapping CUIs to concept information.

    Returns:
        None
    """
    log.info(f"Loading concepts from {concepts_file}")

    with open(concepts_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in tqdm(reader):
            try:
                cui, language, term, source, sab, tty = row[0], row[1], row[14], row[11], row[4], row[12]
                G.add_node(cui, term=term, language=language, source=source, sab=sab, tty=tty)
                cui_to_concept[cui] = {"term": term, "language": language, "source": source, "sab": sab, "tty": tty}
            except Exception as e:
                log.error(f"Error processing concept {row}: {e}")
                raise ValueError(f"Error processing concept {row}: {e}")
