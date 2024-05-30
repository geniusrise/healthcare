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


def process_attributes_file(attributes_file: str, G: nx.DiGraph) -> None:
    """
    Processes the RxNorm attributes file (RXNSAT.RRF) and adds the attributes to the graph.

    Args:
        attributes_file (str): Path to the RxNorm attributes file (RXNSAT.RRF).
        G (nx.DiGraph): The NetworkX graph to which the attributes will be added.

    Returns:
        None
    """
    log.info(f"Loading attributes from {attributes_file}")

    rows = read_rrf_file(attributes_file)
    for row in tqdm(rows):
        try:
            rxcui, lui, sui, rxaui, stype, code, atui, satui, atn, sab, atv, suppress, cvf = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
                row[9],
                row[10],
                row[11],
                row[12],
            )
            if rxcui in G:
                if "attributes" not in G.nodes[rxcui]:
                    G.nodes[rxcui]["attributes"] = []
                G.nodes[rxcui]["attributes"].append(
                    {
                        "lui": lui,
                        "sui": sui,
                        "rxaui": rxaui,
                        "stype": stype,
                        "code": code,
                        "atui": atui,
                        "satui": satui,
                        "atn": atn,
                        "sab": sab,
                        "atv": atv,
                        "suppress": suppress,
                        "cvf": cvf,
                    }
                )
        except Exception as e:
            log.error(f"Error processing attribute {row}: {e}")
            raise ValueError(f"Error processing attribute {row}: {e}")
