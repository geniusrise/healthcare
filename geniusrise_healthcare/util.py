import textwrap
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def calculate_luminance(color):
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]


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


def get_node_levels(G, root_nodes):
    """Get levels of all nodes based on the shortest distance from the root nodes."""
    levels = {}
    visited = set()
    queue = deque([(node, 0) for node in root_nodes])

    while queue:
        current_node, level = queue.popleft()

        if current_node not in visited:
            visited.add(current_node)
            levels[current_node] = level

            for neighbor in G.successors(current_node):
                queue.append((neighbor, level + 1))

    return levels


def draw_subgraph(subgraph, concept_id_to_concept, save_location, highlight_nodes=None, layout_name="arf"):
    num_nodes = len(subgraph.nodes())

    # Decide figure size based on the number of nodes
    fig_size = max(10, int(num_nodes**0.5))

    layouts = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "random": nx.random_layout,
        "shell": nx.shell_layout,
        "spectral": nx.spectral_layout,
        "planar": nx.planar_layout,
        "fruchterman_reingold": nx.fruchterman_reingold_layout,
        "spiral": nx.spiral_layout,
        "multipartite": nx.multipartite_layout,
        "arf": nx.arf_layout,
    }

    layout_func = layouts[layout_name]
    plt.figure(figsize=(fig_size, fig_size), dpi=300)
    labels = {node: concept_id_to_concept.get(str(node), str(node)) for node in subgraph.nodes()}

    if layout_name == "kamada_kawai":
        pos = layout_func(subgraph, weight="degrees")
        node_colors = plt.cm.tab20c(np.linspace(0, 1, len(subgraph.nodes())))
        edge_colors = plt.cm.tab20c(np.linspace(0, 1, len(subgraph.edges())))
    elif layout_name == "shell":
        pos = layout_func(subgraph, scale=5)
        node_colors = plt.cm.tab20c(np.linspace(0, 1, len(subgraph.nodes())))
        edge_colors = plt.cm.tab20c(np.linspace(0, 1, len(subgraph.edges())))
    else:
        pos = layout_func(subgraph)
        # Generate a list of unique colors for nodes and edges
        node_colors = plt.cm.tab20c(np.linspace(0, 1, len(subgraph.nodes())))
        edge_colors = plt.cm.tab20c(np.linspace(0, 1, len(subgraph.edges())))

    node_colors = node_colors.tolist()

    # Get node levels
    if highlight_nodes:
        root_nodes = [n for n in highlight_nodes if n in subgraph.nodes]
        node_levels = get_node_levels(subgraph, root_nodes)

        # Initialize node colors based on levels
        unique_levels = sorted(set(node_levels.values()))
        cmap = plt.cm.get_cmap("tab20c", len(unique_levels))
        node_colors = [
            cmap(unique_levels.index(node_levels.get(node, None)))
            if node_levels.get(node, None) is not None
            else (0, 0, 0, 1)
            for node in subgraph.nodes()
        ]

        highlight_color = plt.cm.Set1(0.0)
        linewidths = [1] * len(subgraph.nodes())
        highlight_indices = [i for i, node in enumerate(subgraph.nodes()) if node in highlight_nodes]

        for i in highlight_indices:
            node_colors[i] = plt.cm.Set1(0.0)
        for i in highlight_indices:
            linewidths[i] = 2  # Set linewidth to 2 for highlighted nodes

    # Create a mapping from node to color
    node_to_color = {node: color for node, color in zip(subgraph.nodes(), node_colors)}

    # Create a list of edge colors based on the origin node's color
    edge_colors = [node_to_color[edge[0]] for edge in subgraph.edges()]

    nx.draw(
        subgraph,
        pos,
        with_labels=False,
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        node_size=600,
        node_shape="o",
        node_color=node_colors,
        linewidths=1,
        font_size=3.0,
        font_color="black",
        edge_color=edge_colors,
        style="solid",
    )

    # Annotate nodes with wrapped text
    for i, (node, (x, y)) in enumerate(pos.items()):
        node_data = subgraph.nodes[node]
        semantic_tag = node_data.get("semantic_tag", "")
        concept_name = concept_id_to_concept.get(str(node), str(node))
        full_label = f"{concept_name}\n({semantic_tag})" if semantic_tag else f"{concept_name}"

        label = textwrap.fill(full_label, width=15)

        # Calculate luminance and decide text color
        luminance = calculate_luminance(node_colors[i][:3])
        text_color = "white" if luminance < 0.5 else "black"

        plt.annotate(
            label,
            xy=(x, y),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=3.0,
            ha="center",
            va="center",
            color=text_color,
        )

    # Label edges
    edge_labels = {}
    for u, v, data in subgraph.edges(data=True):
        relationship_type = concept_id_to_concept.get(str(data.get("relationship_type", "")), "")
        relationship_group = concept_id_to_concept.get(str(data.get("relationship_group", "")), "")
        edge_labels[(u, v)] = f"{relationship_type}\n{relationship_group}"

    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=3.0)

    plt.savefig(f"{save_location}.png", bbox_inches="tight", pad_inches=0.1)
