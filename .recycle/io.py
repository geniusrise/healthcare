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
import pickle
from typing import Dict, Optional, Union

import networkx as nx

import faiss

log = logging.getLogger(__name__)


def load_networkx_graph(file_path: str) -> nx.DiGraph:
    """
    Loads a NetworkX graph from a file.

    Parameters:
    - file_path (str): The file path from which to load the graph.

    Returns:
    nx.DiGraph: The loaded NetworkX graph.
    """
    logging.info(f"Loading NetworkX graph from {file_path}")
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    logging.debug(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    return G


def load_networkx_graph_with_pagerank(file_path: str) -> nx.DiGraph:
    """
    Loads a NetworkX graph from a file and calculates PageRank for each node.

    Parameters:
    - file_path (str): The file path from which to load the graph.

    Returns:
    nx.DiGraph: The loaded NetworkX graph with PageRank as node attributes.
    """
    logging.info(f"Loading NetworkX graph from {file_path}")
    with open(file_path, "rb") as f:
        G = pickle.load(f)

    logging.debug(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")

    logging.info("Calculating PageRank for nodes in the graph.")
    pagerank_dict = nx.pagerank(G)

    # Storing pagerank values as node attributes
    for node, rank in pagerank_dict.items():
        G.nodes[node]["pagerank"] = rank

    logging.debug("PageRank calculation complete and values stored as node attributes.")

    return G


def load_networkx_graph_with_pagerank_snomed(
    file_path: str,
    relationship_weights: Dict[str, float],
    alpha: float = 0.85,
    personalization: Optional[Dict] = None,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    nstart: Optional[Dict] = None,
    weight_key: str = "weight",
    dangling: Optional[Dict] = None,
) -> nx.DiGraph:
    """
    Loads a NetworkX graph from a file and calculates PageRank for each node.

    Parameters:
    - file_path (str): The file path from which to load the graph.
    - relationship_weights (Dict[str, float]): The weights for different relationship types.
    - alpha (float): The damping parameter for PageRank, default is 0.85.
    - personalization (dict): The "personalization vector" consisting of a dictionary with a key for every graph node
                              and nonzero personalization value for each node.
    - max_iter (int): Maximum number of iterations in power method eigenvalue solver.
    - tol (float): Error tolerance used to check convergence in power method solver.
    - nstart (dict): Starting value of PageRank iteration for each node.
    - weight_key (str): Edge data key to use as weight. If None weights are set to 1.
    - dangling (dict): The outedges to be assigned to any “dangling” nodes, i.e., nodes without any outedges.
                       The dict key is the node the outedge points to and the dict value is the weight of that outedge.

    Returns:
    nx.DiGraph: The loaded NetworkX graph with an additional node attribute 'pagerank' storing the PageRank value.
    """
    logging.info(f"Loading NetworkX graph from {file_path}")
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    logging.debug(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")

    # Calculate PageRank
    logging.info("Calculating PageRank for the graph.")
    pagerank = nx.pagerank(
        G,
        alpha=alpha,
        personalization=personalization,
        max_iter=max_iter,
        tol=tol,
        nstart=nstart,
        weight=weight_key,
        dangling=dangling,
    )

    # Store PageRank values as node attributes
    nx.set_node_attributes(G, pagerank, "pagerank")
    logging.debug("PageRank calculation completed and values stored as node attributes.")

    # Update PageRank values based on relationship type weightings
    for node in G.nodes():
        weighted_pagerank = G.nodes[node]["pagerank"]
        for neighbor in G.neighbors(node):
            edge_type = G[node][neighbor].get("relationship_type", "")
            weight = relationship_weights.get(edge_type, 1.0)  # type: ignore
            weighted_pagerank *= weight
        G.nodes[node]["weighted_pagerank"] = weighted_pagerank

    logging.debug("Weighted PageRank calculation based on relationship weights completed.")

    return G


def load_faiss_index(file_path: str, use_cuda: bool = False) -> Union[faiss.Index, faiss.IndexIDMap]:  # type: ignore
    """
    Loads a FAISS index from a file.

    Parameters:
    - file_path (str): The file path from which to load the FAISS index.
    - use_cuda (bool, optional): Whether to load the FAISS index on GPU. Default is False.

    Returns:
    Union[faiss.Index, faiss.IndexIDMap]: The loaded FAISS index.
    """
    log.info(f"Loading FAISS index from {file_path}")

    faiss_index = faiss.read_index(file_path)  # type: ignore

    if use_cuda:
        log.info("Moving FAISS index to GPU.")
        res = faiss.StandardGpuResources()  # type: ignore
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)  # type: ignore

    log.debug(f"Loaded FAISS index with {faiss_index.ntotal} total vectors.")
    return faiss_index


def load_concept_dict(file_path: str) -> Dict[str, str]:
    """
    Loads the description_id_to_concept dictionary from a file.

    Parameters:
    - file_path (str): The file path from where the dictionary will be loaded.

    Returns:
    Dict[str, str]: The loaded description_id_to_concept dictionary.
    """
    logging.info(f"Loading description_id_to_concept dictionary from {file_path}")
    with open(file_path, "rb") as f:
        description_id_to_concept = pickle.load(f)
    return description_id_to_concept


def save_networkx_graph(G: nx.DiGraph, file_path: str) -> None:
    """
    Saves a NetworkX graph to a file.

    Parameters:
    - G (nx.DiGraph): The NetworkX graph to save.
    - file_path (str): The file path where the graph will be saved.

    Returns:
    None
    """
    log.info(f"Saving NetworkX graph to {file_path}")
    with open(file_path, "wb") as f:
        pickle.dump(G, f)


def save_faiss_index(faiss_index: faiss.IndexIDMap, file_path: str) -> None:  # type: ignore
    """
    Saves a FAISS index to a file.

    Parameters:
    - faiss_index (faiss.IndexIDMap): The FAISS index to save.
    - file_path (str): The file path where the FAISS index will be saved.

    Returns:
    None
    """
    log.info(f"Saving FAISS index to {file_path}")
    faiss.write_index(faiss_index, file_path)  # type: ignore


def save_concept_dict(description_id_to_concept: Dict[str, str], file_path: str) -> None:
    """
    Saves the description_id_to_concept dictionary to a file.

    Parameters:
    - description_id_to_concept (Dict[str, str]): The dictionary mapping type IDs to types.
    - file_path (str): The file path where the dictionary will be saved.

    Returns:
    None
    """
    log.info(f"Saving description_id_to_concept dictionary to {file_path}")
    with open(file_path, "wb") as f:
        pickle.dump(description_id_to_concept, f)
