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
import networkx as nx
from .terms import process_terms
from .relationships import process_relationships
from .attributes import process_attributes

log = logging.getLogger(__name__)


def load_loinc(loinc_path: str) -> nx.DiGraph:
    """
    Loads LOINC data into a NetworkX graph.

    Args:
        loinc_path (str): Path to the directory containing the LOINC CSV files.

    Returns:
        The NetworkX graph containing LOINC data.
    """
    G = nx.DiGraph()

    terms_file = os.path.join(loinc_path, "LOINC.csv")
    relationships_file = os.path.join(loinc_path, "LOINC_relations.csv")
    attributes_file = os.path.join(loinc_path, "LOINC_attributes.csv")

    process_terms(terms_file, G)
    process_relationships(relationships_file, G)
    process_attributes(attributes_file, G)

    log.info(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    return G
