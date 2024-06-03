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
from .definitions import process_definitions_file
from .relationships import process_relationships_file
from .semantic_network import process_semantic_network_files
from .semantic_types import process_semantic_types_file
from .attributes import process_attributes_file
from .languages import process_languages_file
from .sources import process_sources_file

log = logging.getLogger(__name__)


def load_umls(umls_path: str) -> Tuple[nx.DiGraph, Dict[str, Dict]]:
    """
    Loads UMLS data into a NetworkX graph.

    Args:
        umls_path (str): Path to the directory containing the UMLS files.

    Returns:
        Tuple containing the NetworkX graph and the source_to_info dictionary.
    """
    G = nx.DiGraph()
    cui_to_concept: Dict[str, Dict] = {}
    source_to_info: Dict[str, Dict] = {}

    concepts_file = os.path.join(umls_path, "MRCONSO.RRF")
    definitions_file = os.path.join(umls_path, "MRDEF.RRF")
    relationships_file = os.path.join(umls_path, "MRREL.RRF")
    semantic_types_file = os.path.join(umls_path, "MRSTY.RRF")
    srdef_file = os.path.join(umls_path, "SRDEF")
    srstr_file = os.path.join(umls_path, "SRSTR")
    attributes_file = os.path.join(umls_path, "MRSAT.RRF")
    sources_file = os.path.join(umls_path, "MRSAB.RRF")

    process_concepts_file(concepts_file, G, cui_to_concept)
    process_definitions_file(definitions_file, G)
    process_relationships_file(relationships_file, G)
    process_semantic_types_file(semantic_types_file, G)
    process_semantic_network_files(srdef_file, srstr_file, G)
    process_attributes_file(attributes_file, G)
    process_languages_file(concepts_file, G)
    process_sources_file(sources_file, source_to_info)

    log.info(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    return G, source_to_info
