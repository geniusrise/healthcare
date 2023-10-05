import logging
from typing import List, Set, Tuple, Union

import networkx as nx
import numpy as np

import faiss
from geniusrise_healthcare.constants import SEMANTIC_TAGS
from geniusrise_healthcare.model import generate_embeddings

log = logging.getLogger(__name__)


def find_adjacent_nodes(
    source_nodes: List[int], G: nx.DiGraph, n: int = 1, top_n: int = 0, undirected: bool = False
) -> List[nx.DiGraph]:
    """
    Finds the subset of nodes that are adjacent to the source nodes within n hops.

    Parameters:
    - source_nodes (List[int]): The source nodes in the NetworkX graph.
    - G (nx.DiGraph): The NetworkX graph.
    - n (int): The number of hops to consider for adjacency.
    - top_n (int): The number of top nodes to consider based on degree.
    - undirected (bool): Whether to consider the graph as undirected.

    Returns:
    List[nx.DiGraph]: A list of subgraphs containing the source nodes, their adjacent nodes, and the edges between them.
    """
    subgraphs = []
    for source_node in source_nodes:
        if not G.has_node(source_node):
            raise ValueError(f"Source node {source_node} not found in the graph.")

        # Find all nodes adjacent to the source node within n hops
        if n == 1:
            subgraph = G.subgraph(set(list(G.predecessors(source_node)) + [source_node]))
        else:
            subgraph = nx.ego_graph(G.reverse(), source_node, radius=n, undirected=undirected, center=True)

        if top_n > 0:
            # Sort nodes by degree and keep only the top_n nodes along with the source node
            sorted_nodes = sorted(subgraph.nodes(), key=lambda x: subgraph.degree(x), reverse=True)
            top_nodes = sorted_nodes[:top_n] if top_n < len(sorted_nodes) else sorted_nodes

            # Always include the source node
            if source_node not in top_nodes:
                top_nodes.append(source_node)
            # Create a new subgraph with only the top_n nodes and the source node
            filtered_subgraph = subgraph.subgraph(top_nodes).copy()
            subgraphs.append(filtered_subgraph)
        else:
            subgraphs.append(subgraph)

    return subgraphs


def find_semantically_similar_nodes(
    faiss_index: faiss.IndexIDMap, embedding: np.ndarray, cutoff_score: float  # type: ignore
) -> List[Tuple[str, float]]:
    """
    Finds the closest nodes to a given node embedding using a FAISS index.

    Parameters:
    - faiss_index (faiss.IndexIDMap): The FAISS index containing the embeddings.
    - embedding (np.ndarray): The vector embedding for which to find the closest nodes.
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
    subgraph = nx.ego_graph(G.to_undirected(), node, radius=n, undirected=True, center=True)
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


def calculate_top_one_percent_nodes(G: nx.DiGraph) -> Set[int]:
    degrees = [(node, degree) for node, degree in G.degree()]
    degrees.sort(key=lambda x: x[1], reverse=True)
    top_one_percent_count = int(len(degrees) * 0.01)
    return set(node for node, _ in degrees[:top_one_percent_count])


def recursive_search(
    G: nx.DiGraph,
    node: int,
    semantic_types: Union[None, str, List[str]],
    stop_at_semantic_types: Union[None, str, List[str]],
    visited: Set[int],
    depth: int,
    max_depth: int,
    current_path: List[int],
    top_one_percent_nodes: Set[int],
) -> List[List[int]]:
    """
    Recursively search the graph starting from a node, collecting all paths leading to nodes of certain semantic types.

    Parameters:
    - G (nx.DiGraph): The NetworkX graph.
    - node (int): The starting node.
    - semantic_types (Union[None, str, List[str]]): The semantic types to collect.
    - stop_at_semantic_types (Union[None, str, List[str]]): The semantic types where the search should stop.
    - visited (Set[int]): Set of visited nodes.
    - depth (int): Current depth of recursion.
    - max_depth (int): Maximum depth allowed for recursion.
    - current_path (List[int]): The current path from the start node.

    Returns:
    List[List[int]]: A list of paths leading to nodes of the specified semantic types.
    """
    if node in visited or depth > max_depth:
        return []

    visited.add(node)
    current_path.append(node)
    paths = []

    node_tag = G.nodes[node].get("tag")
    if node_tag in (stop_at_semantic_types if isinstance(stop_at_semantic_types, list) else [stop_at_semantic_types]):  # type: ignore
        paths.append(current_path.copy())
        current_path.pop()
        return paths

    # Check if the node is in the top 1% of highly connected nodes
    if node in top_one_percent_nodes:
        paths.append(current_path.copy())
        current_path.pop()
        return paths

    if not semantic_types or node_tag in (semantic_types if isinstance(semantic_types, list) else [semantic_types]):
        paths.append(current_path.copy())

    candidates = list(G.predecessors(node)) + list(G.successors(node))
    for neighbor in candidates:
        paths += recursive_search(
            G,
            neighbor,
            semantic_types,
            stop_at_semantic_types,
            visited,
            depth + 1,
            max_depth,
            current_path,
            top_one_percent_nodes=top_one_percent_nodes,
        )

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
    semantic_types: Union[None, str, List[str]] = None,
    stop_at_semantic_types: Union[None, str, List[str]] = None,
    max_depth: int = 3,
) -> Tuple[nx.Graph, List[int]]:
    """
    Finds a consistent set of related conditions, diseases, etc., from the SNOMED graph based on the user's query.

    Parameters:
    - user_terms (List[str]): The user's query terms.
    - G (nx.DiGraph): The NetworkX graph.
    - faiss_index (faiss.IndexIDMap): The FAISS index.
    - model: The model for generating embeddings.
    - tokenizer: The tokenizer for the model.
    - concept_id_to_concept (dict): Dictionary mapping concept IDs to their corresponding concepts.
    - cutoff_score (float, optional): Minimum similarity score to consider a node as similar. Default is 0.1.
    - semantic_types (Union[None, str, List[str]], optional): The semantic types to collect. Default is None.
    - stop_at_semantic_types (Union[None, str, List[str]], optional): The semantic types where the search should stop. Default is None.
    - max_depth (int, optional): Maximum depth for recursive search. Default is 3.

    Returns:
    Tuple[nx.Graph, List[int]]: A tuple containing the resulting subgraph and a list of semantically similar nodes.
    """
    top_1 = calculate_top_one_percent_nodes(G)
    semantically_similar_nodes = []
    if semantic_types and not all(
        st in SEMANTIC_TAGS for st in (semantic_types if isinstance(semantic_types, list) else [semantic_types])
    ):
        raise ValueError(f"Semantic types {semantic_types} not supported")

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
        if stop_at_semantic_types:
            result_paths += recursive_search(
                G,
                node,
                semantic_types=semantic_types,
                stop_at_semantic_types=stop_at_semantic_types,
                visited=visited,
                depth=0,
                max_depth=max_depth,
                current_path=[],
                top_one_percent_nodes=top_1,
            )
        else:
            result_paths += recursive_search(
                G,
                node,
                semantic_types=semantic_types,
                stop_at_semantic_types=None,
                visited=visited,
                depth=0,
                max_depth=max_depth,
                current_path=[],
                top_one_percent_nodes=top_1,
            )

    # Filter out paths that don't contain any of the semantically similar nodes
    filtered_result_paths = [path for path in result_paths if any(node in semantically_similar_nodes for node in path)]

    # Initialize a new directed graph to store the result paths
    result_graph = nx.DiGraph()

    # Add the edges from each filtered path to the result graph
    for path in filtered_result_paths:
        for i in range(len(path) - 1):
            result_graph.add_edge(path[i], path[i + 1])

    log.info(f"Result graph has: {result_graph.number_of_nodes()} nodes and {result_graph.number_of_edges()} edges")
    return result_graph, semantically_similar_nodes
