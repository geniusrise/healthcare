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
import os
import networkx as nx
from .drugs import process_drugs
from .interactions import process_interactions
from .targets import process_targets
from .enzymes import process_enzymes
from .carriers import process_carriers
from .transporters import process_transporters

log = logging.getLogger(__name__)


def load_drugbank(drugbank_path: str) -> nx.DiGraph:
    """
    Loads DrugBank data into a NetworkX graph.

    Args:
        drugbank_path (str): Path to the DrugBank XML file.

    Returns:
        The NetworkX graph containing DrugBank data.
    """
    G = nx.DiGraph()

    drugs_file = os.path.join(drugbank_path, "drugbank.xml")

    process_drugs(drugs_file, G)
    process_interactions(drugs_file, G)
    process_targets(drugs_file, G)
    process_enzymes(drugs_file, G)
    process_carriers(drugs_file, G)
    process_transporters(drugs_file, G)

    log.info(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    return G
