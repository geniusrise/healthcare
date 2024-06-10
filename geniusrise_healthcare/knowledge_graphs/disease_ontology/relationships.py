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
from .utils import read_ontology_file

log = logging.getLogger(__name__)


def process_relationships(ontology_file: str, G: nx.DiGraph) -> None:
    """
    Processes the Disease Ontology file and adds relationships to the graph.

    Args:
        ontology_file (str): Path to the Disease Ontology OBO or OWL file.
        G (nx.DiGraph): The NetworkX graph to which the relationships will be added.

    Returns:
        None
    """
    log.info(f"Loading relationships from {ontology_file}")

    ontology = read_ontology_file(ontology_file)
    for term in ontology.terms():
        if term.id.startswith("DOID:"):
            # Process parent relationships (rdfs:subClassOf)
            for parent in term.superclasses(with_self=False):
                if parent.id.startswith("DOID:"):
                    G.add_edge(term.id, parent.id, type="is_a")

            # Process other relationships (owl:Restriction)
            if hasattr(term, "restrictions"):
                for restriction in term.restrictions():
                    if restriction.property.id.startswith("http://purl.obolibrary.org/obo/"):
                        for target in restriction.value:
                            if target.id.startswith("http://purl.obolibrary.org/obo/"):
                                G.add_edge(term.id, target.id, type=restriction.property.id)
