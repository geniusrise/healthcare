import logging
import pickle
from typing import Dict

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


def load_faiss_index(file_path: str) -> faiss.IndexIDMap:  # type: ignore
    """
    Loads a FAISS index from a file.

    Parameters:
    - file_path (str): The file path from which to load the FAISS index.

    Returns:
    faiss.IndexIDMap: The loaded FAISS index.
    """
    logging.info(f"Loading FAISS index from {file_path}")
    faiss_index = faiss.read_index(file_path)  # type: ignore
    logging.debug(f"Loaded FAISS index with {faiss_index.ntotal} total vectors.")
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
