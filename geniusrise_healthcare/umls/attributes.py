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

# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import networkx as nx
from tqdm import tqdm
from .utils import read_rrf_file

log = logging.getLogger(__name__)


def process_attributes_file(attributes_file: str, G: nx.DiGraph) -> None:
    """
    Processes the UMLS concept attributes file (MRSAT.RRF) and adds the attributes to the graph.

    Args:
        attributes_file (str): Path to the UMLS concept attributes file (MRSAT.RRF).
        G (nx.DiGraph): The NetworkX graph to which the attributes will be added.

    Returns:
        None
    """
    log.info(f"Loading attributes from {attributes_file}")

    rows = read_rrf_file(attributes_file)
    for row in tqdm(rows):
        try:
            cui, atn, atv = row[0], row[4], row[5]
            if cui in G:
                if "attributes" not in G.nodes[cui]:
                    G.nodes[cui]["attributes"] = {}
                G.nodes[cui]["attributes"][atn] = atv
        except Exception as e:
            log.error(f"Error processing attribute {row}: {e}")
            raise ValueError(f"Error processing attribute {row}: {e}")
