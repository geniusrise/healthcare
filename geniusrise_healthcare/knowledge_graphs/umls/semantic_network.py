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
from tqdm import tqdm
from .utils import read_rrf_file

log = logging.getLogger(__name__)


def process_semantic_network_files(srdef_file: str, srstr_file: str, G: nx.DiGraph) -> None:
    """
    Processes the UMLS Semantic Network files (SRDEF and SRSTR) and adds the semantic types and relationships to the graph.

    Args:
        srdef_file (str): Path to the UMLS semantic type definitions file (SRDEF).
        srstr_file (str): Path to the UMLS semantic relationship file (SRSTR).
        G (nx.DiGraph): The NetworkX graph to which the semantic network will be added.

    Returns:
        None
    """
    log.info(f"Loading semantic network information from {srdef_file} and {srstr_file}")

    # Process SRDEF file
    rows = read_rrf_file(srdef_file)
    for row in tqdm(rows):
        try:
            ui, name, description = row[0], row[2], row[6]
            if ui.startswith("T"):  # It's a Semantic Type
                for node, data in G.nodes(data=True):
                    if "semantic_types" in data:
                        for st in data["semantic_types"]:
                            if st["tui"] == ui:
                                st.update({"name": name, "description": description})
            elif ui.startswith("R"):  # It's a Relation
                G.graph["relations"] = G.graph.get("relations", {})
                G.graph["relations"][ui] = {"name": name, "description": description}
        except Exception as e:
            log.error(f"Error processing semantic network definition {row}: {e}")
            raise ValueError(f"Error processing semantic network definition {row}: {e}")

    # Process SRSTR file
    rows = read_rrf_file(srstr_file)
    for row in tqdm(rows):
        try:
            ui1, rel, ui2 = row[0], row[1], row[2]
            G.graph["semantic_network"] = G.graph.get("semantic_network", [])
            G.graph["semantic_network"].append({"source": ui1, "relation": rel, "target": ui2})
        except Exception as e:
            log.error(f"Error processing semantic network relationship {row}: {e}")
            raise ValueError(f"Error processing semantic network relationship {row}: {e}")
