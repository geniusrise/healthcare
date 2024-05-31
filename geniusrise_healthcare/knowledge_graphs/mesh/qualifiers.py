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


def process_qualifiers(qualifiers_file: str, G: nx.DiGraph) -> None:
    """
    Processes the MeSH qualifiers data and adds it to the graph.

    Args:
        qualifiers_file (str): Path to the MeSH qualifiers XML file.
        G (nx.DiGraph): The NetworkX graph to which the qualifiers data will be added.

    Returns:
        None
    """
    log.info(f"Loading qualifiers from {qualifiers_file}")

    tree = read_xml_file(qualifiers_file)
    qualifier_elements = get_elements_by_tag(tree, "QualifierRecord")

    for qualifier in qualifier_elements:
        try:
            qualifier_ui = qualifier.findtext("QualifierUI")
            name = qualifier.findtext("QualifierName/String")
            if qualifier_ui:
                G.add_node(qualifier_ui, name=name, type="qualifier")

                # Process tree numbers
                tree_numbers = qualifier.findall("TreeNumberList/TreeNumber")
                for tree_number in tree_numbers:
                    G.add_node(tree_number.text, type="tree_number")
                    G.add_edge(qualifier_ui, tree_number.text, type="has_tree_number")

        except Exception as e:
            log.error(f"Error processing qualifier {qualifier}: {e}")
            raise ValueError(f"Error processing qualifier {qualifier}: {e}")
