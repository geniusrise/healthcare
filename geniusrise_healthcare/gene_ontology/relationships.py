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

# relationships.py
import logging
import networkx as nx
from .utils import read_ontology_file

log = logging.getLogger(__name__)


def process_relationships(ontology_file: str, G: nx.DiGraph) -> None:
    """
    Processes the Gene Ontology relationships data and adds it to the graph.

    Args:
        ontology_file (str): Path to the Gene Ontology OBO or OWL file.
        G (nx.DiGraph): The NetworkX graph to which the relationships data will be added.

    Returns:
        None
    """
    log.info(f"Loading relationships from {ontology_file}")

    ontology = read_ontology_file(ontology_file)
    for term in ontology.terms():
        try:
            if term.id.startswith("GO:"):
                for parent in term.parents:
                    if parent.id.startswith("GO:"):
                        G.add_edge(term.id, parent.id, type="is_a")
                for relationship in term.relations:
                    for related_term in term.relations[relationship]:
                        if related_term.id.startswith("GO:"):
                            G.add_edge(term.id, related_term.id, type=relationship)
        except Exception as e:
            log.error(f"Error processing relationships for {term}: {e}")
            raise ValueError(f"Error processing relationships for {term}: {e}")
