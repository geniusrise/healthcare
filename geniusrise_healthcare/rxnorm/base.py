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
import os
from typing import Dict, Tuple
import networkx as nx
from .concepts import process_concepts_file
from .relationships import process_relationships_file
from .attributes import process_attributes_file
from .sources import process_sources_file
from .semantic_types import process_semantic_types_file

log = logging.getLogger(__name__)


def load_rxnorm_into_networkx(rxnorm_path: str, version: str = "2022AA") -> Tuple[nx.DiGraph, Dict[str, Dict]]:
    """
    Loads RxNorm data into a NetworkX graph.

    Args:
        rxnorm_path (str): Path to the directory containing the RxNorm files.
        version (str): Version of the RxNorm release (default is "2022AA").

    Returns:
        Tuple containing the NetworkX graph and the source_to_info dictionary.
    """
    G = nx.DiGraph()
    rxcui_to_concept: Dict[str, Dict] = {}
    source_to_info: Dict[str, Dict] = {}

    concepts_file = os.path.join(rxnorm_path, "RXNCONSO.RRF")
    relationships_file = os.path.join(rxnorm_path, "RXNREL.RRF")
    attributes_file = os.path.join(rxnorm_path, "RXNSAT.RRF")
    semantic_types_file = os.path.join(rxnorm_path, "RXNSTY.RRF")
    sources_file = os.path.join(rxnorm_path, "RXNSAB.RRF")

    process_concepts_file(concepts_file, G, rxcui_to_concept)
    process_relationships_file(relationships_file, G)
    process_attributes_file(attributes_file, G)
    process_sources_file(sources_file, source_to_info)
    process_semantic_types_file(semantic_types_file, G)

    log.info(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    return G, source_to_info
