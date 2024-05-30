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

# descriptors.py
import logging
import networkx as nx
from .utils import read_xml_file, get_elements_by_tag

log = logging.getLogger(__name__)


def process_descriptors(descriptors_file: str, G: nx.DiGraph) -> None:
    """
    Processes the MeSH descriptors data and adds it to the graph.

    Args:
        descriptors_file (str): Path to the MeSH descriptors XML file.
        G (nx.DiGraph): The NetworkX graph to which the descriptors data will be added.

    Returns:
        None
    """
    log.info(f"Loading descriptors from {descriptors_file}")

    tree = read_xml_file(descriptors_file)
    descriptor_elements = get_elements_by_tag(tree, "DescriptorRecord")

    for descriptor in descriptor_elements:
        try:
            descriptor_ui = descriptor.findtext("DescriptorUI")
            name = descriptor.findtext("DescriptorName/String")
            if descriptor_ui:
                G.add_node(descriptor_ui, name=name, type="descriptor")

                # Process tree numbers
                tree_numbers = descriptor.findall("TreeNumberList/TreeNumber")
                for tree_number in tree_numbers:
                    G.add_node(tree_number.text, type="tree_number")
                    G.add_edge(descriptor_ui, tree_number.text, type="has_tree_number")

                # Process concept relations
                concepts = descriptor.findall("ConceptList/Concept")
                for concept in concepts:
                    concept_ui = concept.findtext("ConceptUI")
                    concept_name = concept.findtext("ConceptName/String")
                    if concept_ui:
                        G.add_node(concept_ui, name=concept_name, type="concept")
                        G.add_edge(descriptor_ui, concept_ui, type="has_concept")

        except Exception as e:
            log.error(f"Error processing descriptor {descriptor}: {e}")
            raise ValueError(f"Error processing descriptor {descriptor}: {e}")
