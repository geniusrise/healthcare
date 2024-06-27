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
import numpy as np
from tqdm import tqdm

from .util import extract_and_remove_semantic_tag

log = logging.getLogger(__name__)


def process_descriptions_file(descriptions_file: str, G: nx.DiGraph) -> None:
    log.info(f"Loading descriptions from {descriptions_file}")

    with open(descriptions_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip header
        for row in tqdm(reader):
            try:
                (
                    id,
                    effective_time,
                    active,
                    module_id,
                    concept_id,
                    language_code,
                    type_id,
                    term,
                    case_significance_id,
                ) = row[:9]
                if active == "1" and concept_id in G:
                    if "descriptions" not in G.nodes[concept_id]:
                        G.nodes[concept_id]["descriptions"] = []
                    G.nodes[concept_id]["descriptions"].append(
                        {
                            "id": id,
                            "term": term,
                            "language_code": language_code,
                            "type_id": type_id,
                            "case_significance_id": case_significance_id,
                        }
                    )
            except Exception as e:
                log.error(f"Error processing description {row}: {e}")
                raise ValueError(f"Error processing description {row}: {e}")
