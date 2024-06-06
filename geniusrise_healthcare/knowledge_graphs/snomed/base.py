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
import os

import networkx as nx

from .concepts import process_concept_file
from .concrete_relationships import process_concrete_values_file
from .refsets import process_refsets_file
from .relationships import process_relationship_file
from .stated_relationships import process_stated_relationship_file
from .text_definitions import process_text_definition_file

log = logging.getLogger(__name__)


def load_snomed(
    G: nx.DiGraph,
    extract_path: str,
    version: str = "INT_20230901",
) -> nx.DiGraph:
    """
    Loads SNOMED CT data into a NetworkX graph.

    Args:
        G: (nx.DiGraph): The networkx graph.
        extract_path (str): Path to the directory containing the SNOMED CT files.
        version (str): Version of the SNOMED CT files (default is "INT_20230901").

    Returns:
        Tuple containing the graph, description_id_to_concept, concept_id_to_concept, and concept_id_to_text_definition mappings.
    """

    concept_file = os.path.join(extract_path, f"sct2_Description_Snapshot-en_{version}.txt")
    relationship_file = os.path.join(extract_path, f"sct2_Relationship_Snapshot_{version}.txt")
    concrete_values_file = os.path.join(extract_path, f"sct2_RelationshipConcreteValues_Snapshot_{version}.txt")
    stated_relationship_file = os.path.join(extract_path, f"sct2_StatedRelationship_Snapshot_{version}.txt")
    text_definition_file = os.path.join(extract_path, f"sct2_TextDefinition_Snapshot-en_{version}.txt")
    refsets_file = os.path.join(extract_path, f"sct2_sRefset_OWLExpressionSnapshot_{version}.txt")

    process_concept_file(concept_file=concept_file, G=G)

    process_relationship_file(relationship_file, G=G)
    process_concrete_values_file(concrete_values_file, G=G)
    process_stated_relationship_file(stated_relationship_file, G=G)
    process_text_definition_file(text_definition_file, G=G)
    process_refsets_file(refsets_file, G=G)

    log.info(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    return G
