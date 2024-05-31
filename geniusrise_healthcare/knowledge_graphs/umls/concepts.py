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
from typing import Dict
import networkx as nx
from tqdm import tqdm
from .utils import read_rrf_file

log = logging.getLogger(__name__)


def process_concepts_file(concepts_file: str, G: nx.DiGraph, cui_to_concept: Dict[str, Dict]) -> None:
    """
    Processes the UMLS concepts file (MRCONSO.RRF) and adds the concepts to the graph.

    Args:
        concepts_file (str): Path to the UMLS concepts file (MRCONSO.RRF).
        G (nx.DiGraph): The NetworkX graph to which the concepts will be added.
        cui_to_concept (Dict[str, Dict]): Dictionary mapping CUIs to concept information.

    Returns:
        None
    """
    log.info(f"Loading concepts from {concepts_file}")

    rows = read_rrf_file(concepts_file)
    for row in tqdm(rows):
        try:
            cui, language, term, source, sab, tty = row[0], row[1], row[14], row[11], row[4], row[12]
            G.add_node(cui, term=term, language=language, source=source, sab=sab, tty=tty)
            cui_to_concept[cui] = {"term": term, "language": language, "source": source, "sab": sab, "tty": tty}
        except Exception as e:
            log.error(f"Error processing concept {row}: {e}")
            raise ValueError(f"Error processing concept {row}: {e}")
