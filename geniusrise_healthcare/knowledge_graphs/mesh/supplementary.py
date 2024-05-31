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
from .utils import read_xml_file, get_elements_by_tag

log = logging.getLogger(__name__)


def process_supplementary(supplementary_file: str, G: nx.DiGraph) -> None:
    """
    Processes the MeSH supplementary concept records data and adds it to the graph.

    Args:
        supplementary_file (str): Path to the MeSH supplementary concept records XML file.
        G (nx.DiGraph): The NetworkX graph to which the supplementary concept records data will be added.

    Returns:
        None
    """
    log.info(f"Loading supplementary concept records from {supplementary_file}")

    tree = read_xml_file(supplementary_file)
    supplementary_elements = get_elements_by_tag(tree, "SupplementalRecord")

    for supplementary in supplementary_elements:
        try:
            supplementary_ui = supplementary.findtext("SupplementalRecordUI")
            name = supplementary.findtext("SupplementalRecordName/String")
            if supplementary_ui:
                G.add_node(supplementary_ui, name=name, type="supplementary")

                # Process concept relations
                concepts = supplementary.findall("ConceptList/Concept")
                for concept in concepts:
                    concept_ui = concept.findtext("ConceptUI")
                    concept_name = concept.findtext("ConceptName/String")
                    if concept_ui:
                        G.add_node(concept_ui, name=concept_name, type="concept")
                        G.add_edge(supplementary_ui, concept_ui, type="has_concept")

        except Exception as e:
            log.error(f"Error processing supplementary concept record {supplementary}: {e}")
