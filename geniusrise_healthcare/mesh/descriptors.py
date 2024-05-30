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
        descriptor_ui = descriptor.findtext("DescriptorUI")
        descriptor_name = descriptor.findtext("DescriptorName/String")
        tree_numbers = [tn.text for tn in descriptor.findall("TreeNumberList/TreeNumber")]
        concepts = [
            {"ui": c.findtext("ConceptUI"), "name": c.findtext("ConceptName/String")}
            for c in descriptor.findall("ConceptList/Concept")
        ]

        if descriptor_ui:
            G.add_node(descriptor_ui, name=descriptor_name, type="descriptor", tree_numbers=tree_numbers)
            for concept in concepts:
                if concept["ui"]:
                    G.add_node(concept["ui"], name=concept["name"], type="concept")
                    G.add_edge(descriptor_ui, concept["ui"], type="has_concept")
