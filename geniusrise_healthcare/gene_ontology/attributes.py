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

# attributes.py
import logging
import networkx as nx
from .utils import read_ontology_file

log = logging.getLogger(__name__)


def process_attributes(ontology_file: str, G: nx.DiGraph) -> None:
    """
    Processes the Gene Ontology attributes data and adds it to the graph.

    Args:
        ontology_file (str): Path to the Gene Ontology OBO or OWL file.
        G (nx.DiGraph): The NetworkX graph to which the attributes data will be added.

    Returns:
        None
    """
    log.info(f"Loading attributes from {ontology_file}")

    ontology = read_ontology_file(ontology_file)
    for term in ontology.terms():
        try:
            if term.id.startswith("GO:"):
                synonyms = [synonym.description for synonym in term.synonyms]
                xrefs = [xref.id for xref in term.xrefs]
                G.nodes[term.id].update({"synonyms": synonyms, "xrefs": xrefs})
        except Exception as e:
            log.error(f"Error processing attributes for {term}: {e}")
            raise ValueError(f"Error processing attributes for {term}: {e}")
