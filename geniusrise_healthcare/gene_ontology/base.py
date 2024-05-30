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
import networkx as nx
from .terms import process_terms
from .relationships import process_relationships
from .attributes import process_attributes

log = logging.getLogger(__name__)


def load_gene_ontology_into_networkx(ontology_path: str) -> nx.DiGraph:
    """
    Loads Gene Ontology data into a NetworkX graph.

    Args:
        ontology_path (str): Path to the Gene Ontology OBO or OWL file.

    Returns:
        The NetworkX graph containing Gene Ontology data.
    """
    G = nx.DiGraph()

    process_terms(ontology_path, G)
    process_relationships(ontology_path, G)
    process_attributes(ontology_path, G)

    log.info(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    return G
