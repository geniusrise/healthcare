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


def load_umls_into_networkx(umls_path: str, version: str = "2022AA") -> Tuple[nx.DiGraph, Dict[str, Dict]]:
    """
    Loads UMLS data into a NetworkX graph.

    Args:
        umls_path (str): Path to the directory containing the UMLS files.
        version (str): Version of the UMLS release (default is "2022AA").

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
