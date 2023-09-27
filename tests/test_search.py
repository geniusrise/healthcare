import networkx as nx
import pytest
from transformers import AutoModel, AutoTokenizer

from geniusrise_healthcare.io import (
    load_concept_dict,
    load_faiss_index,
    load_networkx_graph,
)
from geniusrise_healthcare.search import (
    find_adjacent_nodes,
    find_related_subgraphs,
    find_semantically_similar_nodes,
)
from geniusrise_healthcare.util import draw_subgraph


@pytest.fixture(scope="module")
def loaded_data():
    G = load_networkx_graph("./saved/snomed.graph")
    faiss_index = load_faiss_index("./saved/faiss.index.old")
    concept_id_to_concept = load_concept_dict("./saved/concept_id_to_concept.pickle")
    description_id_to_concept = load_concept_dict("./saved/description_id_to_concept.pickle")
    return G, faiss_index, concept_id_to_concept, description_id_to_concept


def test_find_semantic_and_adjacent_nodes_compose(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    nodes = ["chest pain", "shortness of breath"]
    subgraphs = []
    for node in nodes:
        # Generate embeddings
        inputs = tokenizer(node, return_tensors="pt", padding=True, truncation=True).to("cpu")
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach()

        closest_nodes = find_semantically_similar_nodes(faiss_index, embeddings, cutoff_score=0.1)

        if len(closest_nodes) > 0:
            inward, outward, neighbors = find_adjacent_nodes([int(x[0]) for x in closest_nodes], G, n=1)
            subgraphs.append(neighbors)
            # draw_subgraph(neighbors, concept_id_to_concept, f"graphs/{' '.join(nodes)}")

    subgraphs = [x for y in subgraphs for x in y if x.number_of_nodes() > 1]
    composed_graph = subgraphs[0].copy()

    # Intersect with each subsequent graph
    for graph in subgraphs[1:]:
        composed_graph = nx.compose(composed_graph, graph)

    # Draw the subgraph
    draw_subgraph(composed_graph, concept_id_to_concept, f"graphs/{' '.join(nodes)}")

    assert composed_graph.number_of_nodes() > 1


def test_find_related_subgraphs(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    user_terms = ["chest pain", "shortness of breath"]

    # Find related terms based on the user's query
    top_subgraphs = find_related_subgraphs(
        user_terms,
        G,
        faiss_index,
        model,
        tokenizer,
        concept_id_to_concept,
        cutoff_score=0.1,
        semantic_type="disorder",
    )
    assert len(top_subgraphs) > 2


def test_find_semantically_similar_nodes(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

    node = "pain in right wrist"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    # Generate embeddings
    inputs = tokenizer(node, return_tensors="pt", padding=True, truncation=True).to("cpu")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach()

    closest_nodes = find_semantically_similar_nodes(faiss_index, embeddings, cutoff_score=0.1)
    assert len(closest_nodes) == 2

    assert isinstance(closest_nodes, list)
    for node, score in closest_nodes:
        assert isinstance(node, str)
        assert isinstance(score, float)
        assert [concept_id_to_concept[str(x[0])] for x in closest_nodes] == [
            "pain in right arm",
            "pain in right upper limb",
        ]
