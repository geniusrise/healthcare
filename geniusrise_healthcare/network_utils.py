import logging

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
