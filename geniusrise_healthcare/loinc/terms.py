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


def process_terms(terms_file: str, G: nx.DiGraph) -> None:
    """
    Processes the LOINC terms data and adds it to the graph.

    Args:
        terms_file (str): Path to the LOINC terms CSV file.
        G (nx.DiGraph): The NetworkX graph to which the terms data will be added.

    Returns:
        None
    """
    log.info(f"Loading terms from {terms_file}")

    rows = read_csv_file(terms_file)
    for row in rows[1:]:  # Skip the header row
        try:
            loinc_num, component, property, time_aspect, system, scale, method, class_type = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
            )
            G.add_node(
                loinc_num,
                component=component,
                property=property,
                time_aspect=time_aspect,
                system=system,
                scale=scale,
                method=method,
                class_type=class_type,
            )
        except Exception as e:
            log.error(f"Error processing term {row}: {e}")
            raise ValueError(f"Error processing term {row}: {e}")
