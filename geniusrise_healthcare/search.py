import logging
from typing import List, Set, Tuple, Optional

import faiss
import networkx as nx
import numpy as np


from geniusrise_healthcare.util import generate_embeddings
from geniusrise_healthcare.constants import SEMANTIC_TAGS

log = logging.getLogger(__name__)


def find_adjacent_nodes(source_nodes: List[int], G: nx.DiGraph, n: int, undirected: bool = False):
    """
    Finds the subset of semantic nodes that are adjacent to the source node within n hops and those that are not.

    Parameters:
    - source_node (str): The source node in the NetworkX graph.
    - semantic_nodes (List[str]): List of semantically close nodes from the FAISS index.
    - G (nx.DiGraph): The NetworkX graph.
    - n (int): The number of hops to consider for adjacency.

    Returns:
    Tuple[Set[str], Set[str]]: Two sets containing the nodes that are both semantic and adjacent, and those that are semantic but not adjacent.
    """
    for source_node in source_nodes:
        if not G.has_node(source_node):
            log.exception(f"Source node {source_node} not found in the graph.")
            raise

    # Find all nodes adjacent to the source node within n hops
    inward = []
    outward = []
    neighbors = []
    for source_node in source_nodes:
        _inward = G.in_edges(source_node)
        _outward = G.out_edges(source_node)
        inward.append(_inward)
        outward.append(_outward)

        subgraph = nx.ego_graph(G, source_node, radius=n, undirected=undirected, center=True)
        neighbors.append(subgraph)

    return inward, outward, neighbors


def find_semantically_similar_nodes(
    faiss_index: faiss.IndexIDMap, embedding: np.ndarray, cutoff_score: float  # type: ignore
) -> List[Tuple[str, float]]:
    """
    Finds the closest nodes to a given node in a NetworkX graph using a FAISS index.

    Parameters:
    - faiss_index (faiss.IndexIDMap): The FAISS index containing the embeddings.
    - G (nx.DiGraph): The NetworkX graph containing SNOMED-CT data.
    - node (np.ndarray): The vector embedding for which to find the closest nodes.
    - cutoff_score (float): The similarity score below which nodes will be ignored.

    Returns:
    List[Tuple[str, float]]: A list of tuples containing the closest node IDs and their similarity scores.
    """
    # Search for the closest nodes in the FAISS index
    distances, closest_node_ids = faiss_index.search(embedding, faiss_index.ntotal)

    # Filter nodes based on the cutoff score and whether they exist in the graph
    closest_nodes = []
    for distance, closest_node_id in zip(distances[0], closest_node_ids[0]):
        similarity_score = 1 / (1 + distance)  # Convert distance to similarity
        closest_node_id_str = str(closest_node_id)
        if similarity_score >= cutoff_score:
            closest_nodes.append((closest_node_id_str, similarity_score))

    log.debug(f"Found {len(closest_nodes)} closest nodes for node.")
    return closest_nodes


def intersect_subgraphs(subgraphs: List[nx.DiGraph]) -> nx.DiGraph:
    """
    Intersects multiple directed graphs and returns a new directed graph containing only the edges
    that exist in all provided graphs.

    Parameters:
    - subgraphs (List[nx.DiGraph]): List of directed graphs to intersect.

    Returns:
    nx.DiGraph: A new directed graph containing only the edges that exist in all provided graphs.
    """
    common_nodes = set(subgraphs[0].nodes())
    for graph in subgraphs[1:]:
        common_nodes.intersection_update(graph.nodes())

    intersection_graph = nx.DiGraph()
    intersection_graph.add_nodes_from(common_nodes)

    for node in common_nodes:
        for graph in subgraphs:
            # Assuming the graphs are directed
            for successor in graph.successors(node):
                if all(successor in g.successors(node) for g in subgraphs):
                    intersection_graph.add_edge(node, successor)

    return intersection_graph


def find_local_important_nodes(G: nx.DiGraph, node: int, n: int = 1) -> List[int]:
    """
    Finds important nodes in the local neighborhood of a given node in a NetworkX graph.

    Parameters:
    - G (nx.DiGraph): The NetworkX graph.
    - node (int): The node for which to find the local important nodes.
    - n (int): The number of hops to consider for local importance.

    Returns:
    List[int]: A list of locally important nodes.
    """
    subgraph = nx.ego_graph(G, node, radius=n, undirected=True, center=True)
    return sorted(subgraph.nodes(), key=lambda x: subgraph.degree(x), reverse=True)


def find_global_important_nodes(G: nx.DiGraph, top_n: int = 10) -> List[int]:
    """
    Finds globally important nodes in a NetworkX graph using PageRank.

    Parameters:
    - G (nx.DiGraph): The NetworkX graph.
    - top_n (int): The number of top nodes to return.

    Returns:
    List[int]: A list of globally important nodes.
    """
    pagerank = nx.pagerank(G)
    return sorted(pagerank, key=pagerank.get, reverse=True)[:top_n]


def recursive_search(
    G: nx.DiGraph,
    node: int,
    semantic_type: Optional[str],
    visited: Set[int],
    depth: int,
    max_depth: int,
    current_path: List[int],
) -> List[List[int]]:
    """
    Recursively search the graph starting from a node, collecting all paths leading to nodes of a certain semantic type.

    Parameters:
    - G (nx.DiGraph): The NetworkX graph.
    - node (int): The starting node.
    - semantic_type (str): The semantic type to collect.
    - visited (Set[int]): Set of visited nodes.
    - depth (int): Current depth of recursion.
    - max_depth (int): Maximum depth allowed for recursion.
    - current_path (List[int]): The current path from the start node.

    Returns:
    List[List[int]]: A list of paths leading to nodes of the specified semantic type.
    """
    if node in visited or depth > max_depth:
        return []

    visited.add(node)
    current_path.append(node)
    paths = []

    if not semantic_type or G.nodes[node].get("tag") == semantic_type:
        paths.append(current_path.copy())

    for neighbor in list(G.predecessors(node)) + list(G.successors(node)):
        paths += recursive_search(G, neighbor, semantic_type, visited, depth + 1, max_depth, current_path)

    current_path.pop()
    return paths


def find_related_subgraphs(
    user_terms: List[str],
    G: nx.DiGraph,
    faiss_index: faiss.IndexIDMap,  # type: ignore
    model,
    tokenizer,
    concept_id_to_concept: dict,
    cutoff_score: float = 0.1,
    semantic_type: Optional[str] = None,
    max_depth: int = 3,
) -> nx.Graph:
    """
    Finds a consistent set of related conditions, diseases, etc., from the SNOMED graph based on the user's query.

    Parameters:
    - user_terms (List[str]): The user's query terms.
    - G (nx.DiGraph): The NetworkX graph.
    - faiss_index (faiss.IndexIDMap): The FAISS index.
    - model: The model for generating embeddings.
    - tokenizer: The tokenizer for the model.
    - max_depth (int): Maximum depth for recursive search.

    Returns:
    nx.Graph: A subgraph containing nodes of the specified semantic type.
    """
    semantically_similar_nodes = []
    if semantic_type and semantic_type not in SEMANTIC_TAGS:
        raise ValueError(f"Semantic type {semantic_type} not supported")

    # query expansion: find semantically similar terms to user's query terms
    for term in user_terms:
        embeddings = generate_embeddings(term, model, tokenizer)
        similar_nodes = find_semantically_similar_nodes(faiss_index, embeddings, cutoff_score=cutoff_score)
        semantically_similar_nodes.extend([int(node) for node, _ in similar_nodes])

    log.info(f"Semantically similar nodes: {semantically_similar_nodes}")

    # Initialize visited set and result paths
    visited = set()  # type: ignore
    result_paths = []

    # Start search from each semantically similar node
    for node in semantically_similar_nodes:
        log.info(f"Processing node {concept_id_to_concept.get(str(node))}")
        result_paths += recursive_search(G, node, semantic_type, visited, depth=0, max_depth=3, current_path=[])

    # Initialize a new directed graph to store the result paths
    result_graph = nx.DiGraph()

    # Add the edges from each path to the result graph
    for path in result_paths:
        for i in range(len(path) - 1):
            result_graph.add_edge(path[i], path[i + 1])

    log.info(f"Result graph has: {result_graph.number_of_nodes()} nodes and {result_graph.number_of_edges()} edges")
    return result_graph
