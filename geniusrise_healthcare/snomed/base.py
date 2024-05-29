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
import os
from typing import Dict, Tuple

import networkx as nx
import torch

from .concepts import process_concept_file
from .concrete_relationships import process_concrete_values_file
from .refsets import process_refsets_file
from .relationships import process_relationship_file
from .stated_relationships import process_stated_relationship_file
from .text_definitions import process_text_definition_file

log = logging.getLogger(__name__)


def load_snomed_into_networkx(
    extract_path: str,
    tokenizer=None,
    model=None,
    faiss_index=None,
    version: str = "INT_20230901",
    use_cuda: bool = True,
    batch_size: int = 10000,
    skip_embedding: bool = False,
) -> Tuple[nx.DiGraph, Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Loads SNOMED CT data into a NetworkX graph.

    Args:
        extract_path (str): Path to the directory containing the SNOMED CT files.
        tokenizer: Tokenizer for processing text data (optional).
        model: Model for generating embeddings (optional).
        faiss_index: Faiss index for embedding storage and search (optional).
        version (str): Version of the SNOMED CT files (default is "INT_20230901").
        use_cuda (bool): Flag to use CUDA for processing (default is True).
        batch_size (int): Batch size for processing embeddings (default is 10000).
        skip_embedding (bool): Flag to skip embedding generation (default is False).

    Returns:
        Tuple containing the graph, description_id_to_concept, concept_id_to_concept, and concept_id_to_text_definition mappings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    G = nx.DiGraph()
    description_id_to_concept: Dict[str, str] = {}
    concept_id_to_concept: Dict[str, str] = {}
    concept_id_to_text_definition: Dict[str, str] = {}

    concept_file = os.path.join(extract_path, f"sct2_Description_Snapshot-en_{version}.txt")
    relationship_file = os.path.join(extract_path, f"sct2_Relationship_Snapshot_{version}.txt")
    concrete_values_file = os.path.join(extract_path, f"sct2_RelationshipConcreteValues_Snapshot_{version}.txt")
    stated_relationship_file = os.path.join(extract_path, f"sct2_StatedRelationship_Snapshot_{version}.txt")
    text_definition_file = os.path.join(extract_path, f"sct2_TextDefinition_Snapshot-en_{version}.txt")
    refsets_file = os.path.join(extract_path, f"sct2_sRefset_OWLExpressionSnapshot_{version}.txt")

    process_concept_file(
        concept_file=concept_file,
        G=G,
        description_id_to_concept=description_id_to_concept,
        concept_id_to_concept=concept_id_to_concept,
        tokenizer=tokenizer,
        model=model,
        faiss_index=faiss_index,
        use_cuda=use_cuda,
        batch_size=batch_size,
        skip_embedding=skip_embedding,
    )

    process_relationship_file(relationship_file, G)
    process_concrete_values_file(concrete_values_file, G)
    process_stated_relationship_file(stated_relationship_file, G)
    process_text_definition_file(text_definition_file, concept_id_to_text_definition)
    process_refsets_file(refsets_file, G)

    log.info(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    return G, description_id_to_concept, concept_id_to_concept, concept_id_to_text_definition
