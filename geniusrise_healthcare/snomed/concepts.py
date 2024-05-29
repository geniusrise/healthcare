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
    use_cuda: bool = True,
    batch_size: int = 10000,
    skip_embedding: bool = False,
) -> None:
    """
    Processes the SNOMED CT concept file and adds concepts to the graph.

    Args:
        concept_file (str): Path to the concept file.
        G (nx.DiGraph): NetworkX graph to which the concepts will be added.
        description_id_to_concept (Dict[str, str]): Dictionary mapping description IDs to concepts.
        concept_id_to_concept (Dict[str, str]): Dictionary mapping concept IDs to concepts.
        tokenizer: Tokenizer for processing text data (optional).
        model: Model for generating embeddings (optional).
        faiss_index: Faiss index for embedding storage and search (optional).
        use_cuda (bool): Flag to use CUDA for processing (default is True).
        batch_size (int): Batch size for processing embeddings (default is 10000).
        skip_embedding (bool): Flag to skip embedding generation (default is False).

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    batch_ids: List[int] = []
    batch_count = 0
    fsns = []
    set_device = False

    csv.field_size_limit(sys.maxsize)

    with open(concept_file, "rbU") as f:
        num_lines = sum(1 for _ in f)

    log.info(f"Loading concepts from {concept_file}")
    with open(concept_file, "r") as f:  # type: ignore
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)  # type: ignore
        next(reader)
        for row in tqdm(reader, total=num_lines):
            try:
                (
                    description_id,
                    active,
                    definition_status,
                    concept_id,
                    language,
                    type_id,
                    concept_name,
                    case_significance,
                ) = (
                    row[0],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                    row[8],
                )

                if active == "1" and language == "en":
                    semantic_tag, fsn_without_tag = extract_and_remove_semantic_tag(concept_name.lower())
                    G.add_node(
                        int(concept_id),
                        type=type_id,
                        tag=semantic_tag,
                        definition_status=definition_status,
                        case_significance=case_significance,
                    )
                    description_id_to_concept[description_id] = fsn_without_tag
                    concept_id_to_concept[concept_id] = fsn_without_tag

                    if not skip_embedding and model and tokenizer and faiss_index:
                        fsns.append(fsn_without_tag)
                        if not set_device:
                            model.to(device)
                            set_device = True

                        batch_ids.append(int(concept_id))
                        batch_count += 1

                        if batch_count >= batch_size:
                            _process_batch(fsns, batch_ids, tokenizer, model, faiss_index, device)
                            batch_ids.clear()
                            fsns.clear()
                            batch_count = 0
            except Exception as e:
                log.error(f"Error processing node {row}: {e}")
                raise ValueError(f"Error processing node {row}: {e}")

    if batch_count > 0 and not skip_embedding and model and tokenizer and faiss_index:
        _process_batch(fsns, batch_ids, tokenizer, model, faiss_index, device)


def _process_batch(fsns: List[str], batch_ids: List[int], tokenizer, model, faiss_index, device) -> None:
    batch_embeddings = []
    for fsn in fsns:
        inputs = tokenizer(fsn, return_tensors="pt").to(device)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach()
        batch_embeddings.append(embeddings)

    log.info("Flushing into faiss")
    batch_embeddings = [x.cpu().numpy() for x in batch_embeddings]
    faiss_index.add_with_ids(np.vstack(batch_embeddings), np.array(batch_ids))
