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


def process_definitions_file(definitions_file: str, G: nx.DiGraph) -> None:
    """
    Processes the UMLS definitions file (MRDEF.RRF) and adds the definitions to the graph.

    Args:
        definitions_file (str): Path to the UMLS definitions file (MRDEF.RRF).
        G (nx.DiGraph): The NetworkX graph to which the definitions will be added.

    Returns:
        None
    """
    log.info(f"Loading definitions from {definitions_file}")

    rows = read_rrf_file(definitions_file)
    for row in tqdm(rows):
        try:
            cui, sab, defn = row[0], row[4], row[5]
            if cui in G:
                G.nodes[cui]["definitions"] = G.nodes[cui].get("definitions", []) + [
                    {"source": sab, "definition": defn}
                ]
        except Exception as e:
            log.error(f"Error processing definition {row}: {e}")
            raise ValueError(f"Error processing definition {row}: {e}")
