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


def process_relationships_file(relationships_file: str, G: nx.DiGraph) -> None:
    """
    Processes the UMLS relationships file (MRREL.RRF) and adds the relationships to the graph.

    Args:
        relationships_file (str): Path to the UMLS relationships file (MRREL.RRF).
        G (nx.DiGraph): The NetworkX graph to which the relationships will be added.

    Returns:
        None
    """
    log.info(f"Loading relationships from {relationships_file}")

    rows = read_rrf_file(relationships_file)
    for row in tqdm(rows):
        try:
            cui1, rel, cui2, rela, sab = row[0], row[3], row[4], row[7], row[10]
            if cui1 in G and cui2 in G:
                G.add_edge(cui1, cui2, rel=rel, rela=rela, sab=sab)
        except Exception as e:
            log.error(f"Error processing relationship {row}: {e}")
            raise ValueError(f"Error processing relationship {row}: {e}")
