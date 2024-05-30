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


def process_transporters(drugbank_file: str, G: nx.DiGraph) -> None:
    """
    Processes the DrugBank transporters data and adds it to the graph.

    Args:
        drugbank_file (str): Path to the DrugBank XML file.
        G (nx.DiGraph): The NetworkX graph to which the transporter data will be added.

    Returns:
        None
    """
    log.info(f"Loading transporters from {drugbank_file}")

    tree = read_xml_file(drugbank_file)
    drug_elements = get_elements_by_tag(tree, "drug")

    for drug in drug_elements:
        drugbank_id = drug.findtext("drugbank-id[@primary='true']")
        transporters = drug.find("transporters")
        if transporters is not None:
            for transporter in transporters.findall("transporter"):
                try:
                    transporter_id = transporter.findtext("id")
                    name = transporter.findtext("name")
                    organism = transporter.findtext("organism")
                    actions = [action.text for action in transporter.findall("actions/action")]
                    if transporter_id:
                        G.add_node(transporter_id, name=name, organism=organism, type="transporter")
                        G.add_edge(drugbank_id, transporter_id, type="transporter", actions=actions)
                except Exception as e:
                    log.error(f"Error processing transporter for drug {drugbank_id}: {e}")
