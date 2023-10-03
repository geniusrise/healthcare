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
    version="INT_20230901",
    use_cuda=True,
    batch_size=10000,
    skip_embedding=False,
) -> Tuple[nx.DiGraph, Dict[str, str], Dict[str, str], Dict[str, str]]:
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
