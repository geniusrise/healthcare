# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import networkx as nx
from tqdm import tqdm
from .utils import read_rrf_file

log = logging.getLogger(__name__)


def process_semantic_types_file(semantic_types_file: str, G: nx.DiGraph) -> None:
    """
    Processes the UMLS semantic types file (MRSTY.RRF) and adds the semantic types to the graph.

    Args:
        semantic_types_file (str): Path to the UMLS semantic types file (MRSTY.RRF).
        G (nx.DiGraph): The NetworkX graph to which the semantic types will be added.

    Returns:
        None
    """
    log.info(f"Loading semantic types from {semantic_types_file}")

    rows = read_rrf_file(semantic_types_file)
    for row in tqdm(rows):
        try:
            cui, tui = row[0], row[1]
            if cui in G:
                if "semantic_types" not in G.nodes[cui]:
                    G.nodes[cui]["semantic_types"] = []
                G.nodes[cui]["semantic_types"].append(tui)
        except Exception as e:
            log.error(f"Error processing semantic type {row}: {e}")
            raise ValueError(f"Error processing semantic type {row}: {e}")
