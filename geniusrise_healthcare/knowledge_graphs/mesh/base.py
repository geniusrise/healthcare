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

# base.py
import logging
import os
import networkx as nx
from .descriptors import process_descriptors
from .qualifiers import process_qualifiers
from .supplementary import process_supplementary

log = logging.getLogger(__name__)


def load_mesh(G: nx.DiGraph, mesh_path: str) -> nx.DiGraph:
    """
    Loads MeSH data into a NetworkX graph.

    Args:
        G: (nx.DiGraph): The networkx graph.
        mesh_path (str): Path to the directory containing the MeSH XML files.

    Returns:
        The NetworkX graph containing MeSH data.
    """

    descriptors_file = os.path.join(mesh_path, "desc2024.xml")
    qualifiers_file = os.path.join(mesh_path, "qual2024.xml")
    supplementary_file = os.path.join(mesh_path, "supp2024.xml")

    process_descriptors(descriptors_file, G)
    process_qualifiers(qualifiers_file, G)
    process_supplementary(supplementary_file, G)

    log.info(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    return G
