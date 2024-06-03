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
from .utils import read_csv_file

log = logging.getLogger(__name__)


def process_relationships(relationships_file: str, G: nx.DiGraph) -> None:
    """
    Processes the LOINC relationships data and adds it to the graph.

    Args:
        relationships_file (str): Path to the LOINC relationships CSV file.
        G (nx.DiGraph): The NetworkX graph to which the relationships data will be added.

    Returns:
        None
    """
    log.info(f"Loading relationships from {relationships_file}")

    rows = read_csv_file(relationships_file)
    for row in rows[1:]:  # Skip the header row
        try:
            loinc_num1, relationship, loinc_num2 = row[0], row[1], row[2]
            G.add_edge(loinc_num1, loinc_num2, relationship=relationship)
        except Exception as e:
            log.error(f"Error processing relationship {row}: {e}")
            raise ValueError(f"Error processing relationship {row}: {e}")