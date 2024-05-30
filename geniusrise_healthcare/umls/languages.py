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


def process_languages_file(languages_file: str, G: nx.DiGraph) -> None:
    """
    Processes the UMLS languages file (MRCONSO.RRF) and adds the language information to the graph.

    Args:
        languages_file (str): Path to the UMLS languages file (MRCONSO.RRF).
        G (nx.DiGraph): The NetworkX graph to which the language information will be added.

    Returns:
        None
    """
    log.info(f"Loading languages from {languages_file}")

    rows = read_rrf_file(languages_file)
    for row in tqdm(rows):
        try:
            cui, language = row[0], row[1]
            if cui in G:
                if "languages" not in G.nodes[cui]:
                    G.nodes[cui]["languages"] = set()
                G.nodes[cui]["languages"].add(language)
        except Exception as e:
            log.error(f"Error processing language {row}: {e}")
            raise ValueError(f"Error processing language {row}: {e}")
