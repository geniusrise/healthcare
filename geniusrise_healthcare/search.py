import logging
import pickle
from typing import Dict, List, Set, Tuple
from collections import Counter

import faiss
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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
        _neighbors = subgraph
        neighbors.append(_neighbors)

    return inward, outward, neighbors


def find_closest_nodes(
    faiss_index: faiss.IndexIDMap, G: nx.DiGraph, node: np.ndarray, cutoff_score: float  # type: ignore
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
    distances, closest_node_ids = faiss_index.search(node, faiss_index.ntotal)

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


def print_dfs_chain(G, node, visited, concept_id_to_concept, chain):
    visited.add(node)
    chain.append(concept_id_to_concept.get(str(node), node))

    for neighbor in G.neighbors(node):
        if neighbor not in visited:
            edge_data = G.get_edge_data(node, neighbor)
            print_dfs_chain(G, neighbor, visited, concept_id_to_concept, chain)
            chain_str = " --> ".join(chain)
            print(
                f"Chain: {chain_str} --({concept_id_to_concept[edge_data['relationship_type']]})---> {concept_id_to_concept.get(str(neighbor), neighbor)}"
            )
            chain.pop()


def draw_subgraph(subgraph, concept_id_to_concept, save_location):
    plt.figure(figsize=(20, 20), dpi=300)
    labels = {node: concept_id_to_concept.get(str(node), str(node)) for node in subgraph.nodes()}
    pos = nx.spring_layout(subgraph, k=0.6, iterations=50)
    nx.draw(
        subgraph,
        pos,
        labels=labels,
        with_labels=True,
        arrows=True,
        node_size=400,
        node_shape="s",
        node_color="lightblue",
        font_size=8.0,
        font_color="black",
        edge_color="coral",
        style="dashed",
    )
    plt.savefig(f"{save_location}.png", bbox_inches="tight", pad_inches=0.1)


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


def rank_nodes(
    candidate_nodes: Set[int], G: nx.DiGraph, faiss_index: faiss.IndexIDMap, user_terms: List[str], model, tokenizer  # type: ignore
) -> List[int]:
    """
    Ranks candidate nodes based on local and global importance and semantic similarity to the user's query.

    Parameters:
    - candidate_nodes (Set[int]): The set of candidate nodes.
    - G (nx.DiGraph): The NetworkX graph.
    - faiss_index (faiss.IndexIDMap): The FAISS index.
    - user_terms (List[str]): The user's query terms.
    - model: The BERT model for generating embeddings.
    - tokenizer: The tokenizer for the BERT model.

    Returns:
    List[int]: A list of ranked nodes.
    """
    local_scores = Counter()  # type: ignore
    global_scores = Counter()  # type: ignore
    semantic_scores = Counter()  # type: ignore

    global_important_nodes = find_global_important_nodes(G)
    for node in candidate_nodes:
        local_scores[node] = len(find_local_important_nodes(G, node))
        global_scores[node] = 1 if node in global_important_nodes else 0

        embeddings = generate_embeddings(str(node), model, tokenizer)
        _, closest_nodes = find_closest_nodes(faiss_index, G, embeddings, cutoff_score=0.1)
        semantic_scores[node] = sum(score for _, score in closest_nodes if node in candidate_nodes)  # type: ignore

    # Combine the scores
    combined_scores = Counter()  # type: ignore
    for node in candidate_nodes:
        combined_scores[node] = local_scores[node] + global_scores[node] + semantic_scores[node]

    return [node for node, _ in combined_scores.most_common()]


def generate_embeddings(term: str, model, tokenizer) -> np.ndarray:
    """
    Generates embeddings for a given term using a BERT model.

    Parameters:
    - term (str): The term for which to generate the embeddings.
    - model: The BERT model.
    - tokenizer: The tokenizer for the BERT model.

    Returns:
    np.ndarray: The generated embeddings.
    """
    inputs = tokenizer(term, return_tensors="pt", padding=True, truncation=True).to("cpu")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings


def find_related_terms(
    user_terms: List[str], G: nx.DiGraph, faiss_index: faiss.IndexIDMap, model, tokenizer  # type: ignore
) -> List[int]:
    """
    Finds a consistent set of related conditions, diseases, etc., from the SNOMED graph based on the user's query.

    Parameters:
    - user_terms (List[str]): The user's query terms.
    - G (nx.DiGraph): The NetworkX graph.
    - faiss_index (faiss.IndexIDMap): The FAISS index.
    - model: The BERT model for generating embeddings.
    - tokenizer: The tokenizer for the BERT model.

    Returns:
    List[int]: A list of related nodes.
    """
    semantically_similar_nodes = []
    for term in user_terms:
        embeddings = generate_embeddings(term, model, tokenizer)
        similar_nodes = find_closest_nodes(faiss_index, G, embeddings, cutoff_score=0.1)
        semantically_similar_nodes.extend([int(node) for node, _ in similar_nodes])

    local_important_nodes = []
    for node in semantically_similar_nodes:
        local_important_nodes.extend(find_local_important_nodes(G, node))

    global_important_nodes = find_global_important_nodes(G)

    candidate_nodes = set(semantically_similar_nodes) & set(local_important_nodes) & set(global_important_nodes)
    ranked_nodes = rank_nodes(candidate_nodes, G, faiss_index, user_terms, model, tokenizer)

    return ranked_nodes


def load_networkx_graph(file_path: str) -> nx.DiGraph:
    """
    Loads a NetworkX graph from a file.

    Parameters:
    - file_path (str): The file path from which to load the graph.

    Returns:
    nx.DiGraph: The loaded NetworkX graph.
    """
    log.info(f"Loading NetworkX graph from {file_path}")
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    log.debug(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into the graph.")
    return G


def load_faiss_index(file_path: str) -> faiss.IndexIDMap:  # type: ignore
    """
    Loads a FAISS index from a file.

    Parameters:
    - file_path (str): The file path from which to load the FAISS index.

    Returns:
    faiss.IndexIDMap: The loaded FAISS index.
    """
    log.info(f"Loading FAISS index from {file_path}")
    faiss_index = faiss.read_index(file_path)  # type: ignore
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
    log.info(f"Loading description_id_to_concept dictionary from {file_path}")
    with open(file_path, "rb") as f:
        description_id_to_concept = pickle.load(f)
    return description_id_to_concept
