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


def process_targets_file(targets_file: str, G: nx.DiGraph) -> None:
    """
    Processes the DrugBank targets data and adds it to the graph.

    Args:
        targets_file (str): Path to the DrugBank XML file.
        G (nx.DiGraph): The NetworkX graph to which the target data will be added.

    Returns:
        None
    """
    log.info(f"Loading targets from {targets_file}")

    tree = read_xml_file(targets_file)
    drug_elements = get_elements_by_tag(tree, "drug")

    for drug in drug_elements:
        try:
            drugbank_id = drug.findtext("drugbank-id[@primary='true']")
            targets = drug.find("targets")
            if targets is not None:
                for target in targets.findall("target"):
                    target_id = target.findtext("id")
                    name = target.findtext("name")
                    organism = target.findtext("organism")
                    if target_id:
                        G.add_node(target_id, name=name, organism=organism, type="target")
                        G.add_edge(drugbank_id, target_id, type="targets")
        except Exception as e:
            log.error(f"Error processing target for drug {drug}: {e}")
            raise ValueError(f"Error processing target for drug {drug}: {e}")
