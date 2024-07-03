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
import sys

import networkx as nx
import numpy as np
from tqdm import tqdm

from .util import extract_and_remove_semantic_tag

log = logging.getLogger(__name__)


def process_concept_file(concept_file: str, G: nx.DiGraph) -> None:
    """
    Processes the SNOMED CT concept file and adds concepts to the graph.

    Args:
        concept_file (str): Path to the concept file.
        G (nx.DiGraph): NetworkX graph to which the concepts will be added.

    Returns:
        None
    """
    csv.field_size_limit(sys.maxsize)

    log.info(f"Loading concepts from {concept_file}")

    with open(concept_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip header
        for row in tqdm(reader):
            try:
                (id, effective_time, active, module_id, definition_status_id) = row[:5]
                if active == "1":
                    G.add_node(
                        id,
                        type="concept",
                        effective_time=effective_time,
                        module_id=module_id,
                        definition_status_id=definition_status_id,
                    )
            except Exception as e:
                log.error(f"Error processing concept {row}: {e}")
                raise ValueError(f"Error processing concept {row}: {e}")
