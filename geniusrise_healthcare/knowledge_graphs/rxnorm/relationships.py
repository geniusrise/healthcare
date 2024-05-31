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


def process_relationships_file(relationships_file: str, G: nx.DiGraph) -> None:
    """
    Processes the RxNorm relationships file (RXNREL.RRF) and adds the relationships to the graph.

    Args:
        relationships_file (str): Path to the RxNorm relationships file (RXNREL.RRF).
        G (nx.DiGraph): The NetworkX graph to which the relationships will be added.

    Returns:
        None
    """
    log.info(f"Loading relationships from {relationships_file}")

    rows = read_rrf_file(relationships_file)
    for row in tqdm(rows):
        try:
            rxcui1, rxaui1, stype1, rel, rxcui2, rxaui2, stype2, rela, rui, sab, suppress = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
                row[10],
                row[16],
            )
            if rxcui1 in G and rxcui2 in G:
                G.add_edge(
                    rxcui1,
                    rxcui2,
                    rxaui1=rxaui1,
                    stype1=stype1,
                    rel=rel,
                    rxaui2=rxaui2,
                    stype2=stype2,
                    rela=rela,
                    rui=rui,
                    sab=sab,
                    suppress=suppress,
                )
        except Exception as e:
            log.error(f"Error processing relationship {row}: {e}")
            raise ValueError(f"Error processing relationship {row}: {e}")
