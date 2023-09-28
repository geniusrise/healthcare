import csv
import logging
import sys
from typing import Dict, List

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from .util import extract_and_remove_semantic_tag

log = logging.getLogger(__name__)


def process_concept_file(
    concept_file: str,
    G: nx.DiGraph,
    description_id_to_concept: Dict[str, str],
    concept_id_to_concept: Dict[str, str],
    tokenizer=None,
    model=None,
    faiss_index=None,
    use_gpu=True,
    batch_size=10000,
    skip_embedding=False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    # Initialize batch variables
    batch_embeddings: List[torch.Tensor] = []
    batch_ids: List[int] = []
    batch_count = 0

    csv.field_size_limit(sys.maxsize)

    file_length = 0
    with open(concept_file, "rbU") as f:
        num_lines = sum(1 for _ in f)

    log.info(f"Loading concepts from {concept_file}")
    with open(concept_file, "r") as f:  # type: ignore
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)  # type: ignore
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

                    # Generate embeddings
                    if not skip_embedding and model and tokenizer and faiss_index:
                        model.to(device)
                        inputs = tokenizer(
                            fsn_without_tag,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        ).to(device)
                        outputs = model(**inputs)
                        embeddings = outputs.last_hidden_state.mean(dim=1).detach()

                        # Add to batch
                        batch_embeddings.append(embeddings)
                        batch_ids.append(int(description_id))
                        batch_count += 1

                        # Process batch if it reaches the batch_size
                        if batch_count >= batch_size:
                            log.info("Flushing into faiss")
                            batch_embeddings = [
                                x.cpu().numpy() if type(x) is not np.ndarray else x for x in batch_embeddings
                            ]
                            faiss_index.add_with_ids(np.vstack(batch_embeddings), np.array(batch_ids))
                            batch_embeddings.clear()
                            batch_ids.clear()
                            batch_count = 0
            except Exception as e:
                raise ValueError(f"Error processing node {row}: {e}")

    # Process remaining batch
    if batch_count > 0 and not skip_embedding and model and tokenizer and faiss_index:
        batch_embeddings = [x.cpu().numpy() if type(x) is not np.ndarray else x for x in batch_embeddings]
        faiss_index.add_with_ids(np.vstack(batch_embeddings), np.array(batch_ids))
