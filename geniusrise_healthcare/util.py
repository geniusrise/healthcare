import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import textwrap


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


def draw_subgraph(subgraph, concept_id_to_concept, save_location, layout_name="arf"):
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
        node_colors = plt.cm.tab20b(np.linspace(0, 1, len(subgraph.nodes())))
        edge_colors = plt.cm.tab20b(np.linspace(0, 1, len(subgraph.edges())))
    elif layout_name == "shell":
        pos = layout_func(subgraph, scale=0.1)
        node_colors = plt.cm.tab20b(np.linspace(0, 1, len(subgraph.nodes())))
        edge_colors = plt.cm.tab20b(np.linspace(0, 1, len(subgraph.edges())))
    else:
        pos = layout_func(subgraph)
        # Generate a list of unique colors for nodes and edges
        node_colors = plt.cm.plasma(np.linspace(0, 1, len(subgraph.nodes())))
        edge_colors = plt.cm.plasma(np.linspace(0, 1, len(subgraph.edges())))

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

    plt.savefig(f"{save_location}.png", bbox_inches="tight", pad_inches=0.1)


def generate_embeddings(term: str, model, tokenizer) -> np.ndarray:
    """
    Generates embeddings for a given term using a BERT model.

    Parameters:
    - term (str): The term for which to generate the embeddings.
    - model: The model.
    - tokenizer: The tokenizer for the model.

    Returns:
    np.ndarray: The generated embeddings.
    """
    inputs = tokenizer(term, return_tensors="pt", padding=True, truncation=True).to("cpu")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings
