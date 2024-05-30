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


def process_carriers_file(carriers_file: str, G: nx.DiGraph) -> None:
    """
    Processes the DrugBank carriers data and adds it to the graph.

    Args:
        carriers_file (str): Path to the DrugBank XML file.
        G (nx.DiGraph): The NetworkX graph to which the carrier data will be added.

    Returns:
        None
    """
    log.info(f"Loading carriers from {carriers_file}")

    tree = read_xml_file(carriers_file)
    drug_elements = get_elements_by_tag(tree, "drug")

    for drug in drug_elements:
        try:
            drugbank_id = drug.findtext("drugbank-id[@primary='true']")
            carriers = drug.find("carriers")
            if carriers is not None:
                for carrier in carriers.findall("carrier"):
                    carrier_id = carrier.findtext("id")
                    name = carrier.findtext("name")
                    organism = carrier.findtext("organism")
                    if carrier_id:
                        G.add_node(carrier_id, name=name, organism=organism, type="carrier")
                        G.add_edge(drugbank_id, carrier_id, type="carriers")
        except Exception as e:
            log.error(f"Error processing carrier for drug {drug}: {e}")
            raise ValueError(f"Error processing carrier for drug {drug}: {e}")
