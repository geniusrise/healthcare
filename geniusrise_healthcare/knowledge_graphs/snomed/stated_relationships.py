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

import csv
import logging

import networkx as nx
from tqdm import tqdm

log = logging.getLogger(__name__)


def process_stated_relationship_file(stated_relationship_file: str, G: nx.DiGraph) -> None:
    """
    Processes the SNOMED CT stated relationship file and adds the relationships to the graph.

    Args:
        stated_relationship_file (str): Path to the stated relationship file.
        G (nx.DiGraph): The NetworkX graph to which the relationships will be added.

    Returns:
        None
    """
    log.info(f"Loading stated relationships from {stated_relationship_file}")

    with open(stated_relationship_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip header
        for row in tqdm(reader):
            try:
                (
                    id,
                    effective_time,
                    active,
                    module_id,
                    source_id,
                    destination_id,
                    relationship_group,
                    type_id,
                    characteristic_type_id,
                    modifier_id,
                ) = row[:10]
                if active == "1" and source_id in G and destination_id in G:
                    G.add_edge(
                        source_id,
                        destination_id,
                        id=id,
                        type="stated_relationship",
                        relationship_group=relationship_group,
                        type_id=type_id,
                        characteristic_type_id=characteristic_type_id,
                        modifier_id=modifier_id,
                    )
            except Exception as e:
                log.error(f"Error processing stated relationship {row}: {e}")
                raise ValueError(f"Error processing stated relationship {row}: {e}")
