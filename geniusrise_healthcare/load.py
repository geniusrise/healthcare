import csv
import logging
import os
import pickle
import zipfile
from typing import Dict, Tuple, List
import sys

import torch
import faiss
import networkx as nx
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


def unzip_snomed_ct(zip_path: str, extract_path: str) -> None:
    """
    Unzips the SNOMED-CT dataset.

    Parameters:
    - zip_path (str): Path to the SNOMED-CT zip file.
    - extract_path (str): Directory where the files will be extracted.

    Returns:
    None
    """
    log.info(f"Unzipping SNOMED-CT dataset from {zip_path} to {extract_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def extract_and_remove_semantic_tag(fsn: str) -> Tuple[str, str]:
    semantic_tag = fsn.split("(")[-1].rstrip(")").strip()
    fsn_without_tag = fsn.rsplit("(", 1)[0].strip()
    semantic_tag = "" if semantic_tag == fsn_without_tag else semantic_tag
    return semantic_tag, fsn_without_tag


def load_snomed_into_networkx(
    extract_path: str,
    tokenizer=None,
    model=None,
    faiss_index=None,
    version="INT_20230901",
    use_gpu=True,
    batch_size=10000,
    skip_embedding=False,
) -> Tuple[nx.DiGraph, Dict[str, str], Dict[str, str]]:
    """
    Loads SNOMED-CT data into a NetworkX directed graph.

    Parameters:
    - extract_path (str): Directory where the SNOMED-CT files are located.
    - tokenizer: Tokenizer for the model.
    - model: Pre-trained model for generating embeddings.
    - faiss_index: FAISS index for storing embeddings.
    - version (str): Version of the SNOMED-CT data.
    - use_gpu (bool): Whether to use GPU for model inference.
    - batch_size (int): The size of each batch for bulk processing.

    Returns:
    Tuple[nx.DiGraph, Dict[str, str], Dict[str, str]]: A NetworkX directed graph and a couple of dictionaries mapping  snomed and concept IDs to concepts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    G = nx.DiGraph()
    description_id_to_concept: Dict[str, str] = {}
    concept_id_to_concept: Dict[str, str] = {}
    concept_file = os.path.join(extract_path, f"sct2_Description_Snapshot-en_{version}.txt")
    relationship_file = os.path.join(extract_path, f"sct2_Relationship_Snapshot_{version}.txt")

    # Initialize batch variables
    batch_embeddings: List[torch.Tensor] = []
    batch_ids: List[int] = []
    batch_count = 0

    csv.field_size_limit(sys.maxsize)

    file_length = 0
    with open(concept_file, "rbU") as f:
        num_lines = sum(1 for _ in f)

    log.info(f"Loading concepts from {concept_file}")
    with open(concept_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)  # Skip header
        for row in tqdm(reader, total=num_lines):
            try:
                description_id, active, concept_id, language, concept_name, type_id = (
                    row[0],
                    row[2],
                    row[4],
                    row[5],
                    row[7],
                    row[6],
                )

                if active == "1" and language == "en":
                    semantic_tag, fsn_without_tag = extract_and_remove_semantic_tag(concept_name.lower())
                    G.add_node(int(concept_id), type=type_id, tag=semantic_tag)
                    description_id_to_concept[description_id] = fsn_without_tag
                    concept_id_to_concept[concept_id] = fsn_without_tag

                    # # Generate embeddings
                    if not skip_embedding and model and tokenizer and faiss_index:
                        model.to(device)
                        inputs = tokenizer(fsn_without_tag, return_tensors="pt", padding=True, truncation=True).to(
                            device
                        )
                        outputs = model(**inputs)
                        embeddings = outputs.last_hidden_state.mean(dim=1).detach()

                        # Add to batch
                        batch_embeddings.append(embeddings)
                        batch_ids.append(int(description_id))
                        batch_count += 1

                        # Process batch if it reaches the batch_size
                        if batch_count >= batch_size:
                            log.info(f"Flushing into faiss")
                            batch_embeddings = [
                                x.cpu().numpy() if type(x) is not np.ndarray else x for x in batch_embeddings
                            ]
                            faiss_index.add_with_ids(np.vstack(batch_embeddings), np.array(batch_ids))
                            batch_embeddings.clear()
                            batch_ids.clear()
                            batch_count = 0
            except Exception as e:
                log.exception("Error processing node {row}: {e}")

    # Process remaining batch
    if batch_count > 0 and not skip_embedding and model and tokenizer and faiss_index:
        batch_embeddings = [x.cpu().numpy() if type(x) is not np.ndarray else x for x in batch_embeddings]
        faiss_index.add_with_ids(np.vstack(batch_embeddings), np.array(batch_ids))

    file_length = 0
    with open(relationship_file, "rbU") as f:
        num_lines = sum(1 for _ in f)

    log.info(f"Loading relationships from {relationship_file}")
    with open(relationship_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)  # Skip header
        for row in tqdm(reader, total=num_lines):
            try:
                source_id, dest_id, active, relationship_type, relationship_group = (
                    row[4],
                    row[5],
                    row[2],
                    row[7],
                    row[6],
                )
                if active == "1":
                    G.add_edge(
                        int(source_id),
                        int(dest_id),
                        relationship_type=relationship_type,
                        relationship_group=relationship_group,
                    )
            except Exception as e:
                log.exception("Error processing relation {row}")

    log.info(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    return G, description_id_to_concept, concept_id_to_concept
