import logging
from typing import List

import networkx as nx

log = logging.getLogger(__name__)


def find_largest_strongly_connected_component(G: nx.DiGraph) -> nx.DiGraph:
    """
    Finds the largest strongly connected component in a directed graph.

    Parameters:
    - G (nx.DiGraph): The directed graph.

    Returns:
    nx.DiGraph: The largest strongly connected component as a new directed graph.
    """
    largest_scc = max(list(nx.strongly_connected_components(G)), key=len)
    largest_scc = G.subgraph(largest_scc).copy()
    log.info(f"Result component has: {largest_scc.number_of_nodes()} nodes and {largest_scc.number_of_edges()} edges")
    return largest_scc


def find_largest_weakly_connected_component(G: nx.DiGraph) -> nx.DiGraph:
    """
    Finds the largest weakly connected component in a directed graph.

    Parameters:
    - G (nx.DiGraph): The directed graph.

    Returns:
    nx.DiGraph: The largest weakly connected component as a new directed graph.
    """
    largest_wcc = max(list(nx.weakly_connected_components(G)), key=len)
    largest_wcc = G.subgraph(largest_wcc).copy()
    log.info(f"Result component has: {largest_wcc.number_of_nodes()} nodes and {largest_wcc.number_of_edges()} edges")
    return largest_wcc


def find_largest_attracting_component(G: nx.DiGraph) -> nx.DiGraph:
    """
    Finds the largest attracting component in a directed graph.

    Parameters:
    - G (nx.DiGraph): The directed graph.

    Returns:
    nx.DiGraph: The largest attracting component as a new directed graph.
    """
    largest_ac = max(list(nx.attracting_components(G)), key=len)
    largest_ac = G.subgraph(largest_ac).copy()
    log.info(f"Result component has: {largest_ac.number_of_nodes()} nodes and {largest_ac.number_of_edges()} edges")
    return largest_ac


def find_largest_connected_component(G: nx.DiGraph) -> nx.DiGraph:
    """
    Finds the largest connected component in a directed graph.

    Parameters:
    - G (nx.DiGraph): The directed graph.

    Returns:
    nx.DiGraph: The largest connected component as a new directed graph.
    """
    largest_cc = max(list(nx.connected_components(G.to_undirected())), key=len)
    largest_cc = G.subgraph(largest_cc).copy()
    log.info(f"Result component has: {largest_cc.number_of_nodes()} nodes and {largest_cc.number_of_edges()} edges")
    return largest_cc


def find_largest_connected_component_with_nodes(G: nx.DiGraph, nodes: List[int]) -> nx.DiGraph:
    """
    Finds the largest connected component in a directed graph that contains the maximum number of given nodes.

    Parameters:
    - G (nx.DiGraph): The directed graph.
    - nodes (List[int]): List of node IDs that should be contained in the connected component.

    Returns:
    nx.DiGraph: The largest connected component containing the maximum number of given nodes as a new directed graph.
    """
    # Convert the graph to undirected for connected component analysis
    G_undirected = G.to_undirected()

    # Find all connected components
    connected_components = list(nx.connected_components(G_undirected))

    # Filter components that contain any of the given nodes
    relevant_components = [comp for comp in connected_components if any(node in comp for node in nodes)]

    if not relevant_components:
        log.info("No connected component contains any of the given nodes.")
        return nx.DiGraph()

    # Find the largest among the filtered components
    largest_relevant_component = max(relevant_components, key=lambda comp: len(set(comp).intersection(nodes)))

    # Create a subgraph for the largest component
    largest_relevant_subgraph = G.subgraph(largest_relevant_component).copy()

    log.info(
        f"Result component has: {largest_relevant_subgraph.number_of_nodes()} nodes and {largest_relevant_subgraph.number_of_edges()} edges"
    )
    return largest_relevant_subgraph
