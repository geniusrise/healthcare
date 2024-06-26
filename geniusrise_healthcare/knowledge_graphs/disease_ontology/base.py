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
from .diseases import process_diseases
from .relationships import process_relationships

log = logging.getLogger(__name__)


def load_disease_ontology(G: nx.DiGraph, ontology_file: str) -> nx.DiGraph:
    """
    Loads Disease Ontology data into a NetworkX graph.

    Args:
        G: (nx.DiGraph): The networkx graph.
        ontology_file (str): Path to the Disease Ontology OBO or OWL file.

    Returns:
        The NetworkX graph containing Disease Ontology data.
    """
    try:
        process_diseases(ontology_file, G)

        process_relationships(ontology_file, G)

        log.info(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    except Exception as e:
        log.error(f"Error loading Disease Ontology data: {e}")
        raise e

    return G
