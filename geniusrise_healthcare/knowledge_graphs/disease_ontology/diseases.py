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


def process_diseases(ontology_file: str, G: nx.DiGraph) -> None:
    """
    Processes the Disease Ontology file and adds disease nodes to the graph.

    Args:
        ontology_file (str): Path to the Disease Ontology OBO or OWL file.
        G (nx.DiGraph): The NetworkX graph to which the disease nodes will be added.

    Returns:
        None
    """
    log.info(f"Loading diseases from {ontology_file}")

    ontology = read_ontology_file(ontology_file)
    for term in ontology.terms():
        try:
            if term.id.startswith("DOID:"):
                synonyms = [synonym.description for synonym in term.synonyms]
                xrefs = [xref.id for xref in term.xrefs]
                G.add_node(
                    term.id, name=term.name, definition=term.definition, type="disease", synonyms=synonyms, xrefs=xrefs
                )
        except Exception as e:
            log.error(f"Error processing disease {term}: {e}")
