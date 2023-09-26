import pytest

from geniusrise_healthcare.io import (
    load_networkx_graph,
    load_faiss_index,
    load_concept_dict,
)
from geniusrise_healthcare.search import (
    find_adjacent_nodes,
    find_closest_nodes,
    draw_subgraph,
    intersect_subgraphs,
    find_related_terms,
)
import torch
from transformers import AutoTokenizer, AutoModel
import networkx as nx


@pytest.fixture(scope="module")
def loaded_data():
    G = load_networkx_graph("./saved/snomed.graph")
    faiss_index = load_faiss_index("./saved/faiss.index.old")
    concept_id_to_concept = load_concept_dict("./saved/concept_id_to_concept.pickle")
    description_id_to_concept = load_concept_dict("./saved/description_id_to_concept.pickle")
    return G, faiss_index, concept_id_to_concept, description_id_to_concept


# def test_find_semantic_and_adjacent_nodes_compose(loaded_data):
#     G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     model = AutoModel.from_pretrained("bert-base-uncased")

#     nodes = ["Swollen ankle", "soft", "pains like needle", "bluish in color", "painful", "impact"]
#     subgraphs = []
#     for node in nodes:
#         # Generate embeddings
#         inputs = tokenizer(node, return_tensors="pt", padding=True, truncation=True).to("cpu")
#         outputs = model(**inputs)
#         embeddings = outputs.last_hidden_state.mean(dim=1).detach()

#         closest_nodes = find_closest_nodes(faiss_index, G, embeddings, cutoff_score=0.1)

#         if len(closest_nodes) > 0:
#             inward, outward, neighbors = find_adjacent_nodes([int(x[0]) for x in closest_nodes], G, n=1)
#             subgraphs.append(neighbors)
#             # draw_subgraph(neighbors, concept_id_to_concept, f"graphs/{' '.join(nodes)}")

#     subgraphs = [x for y in subgraphs for x in y if x.number_of_nodes() > 1]
#     composed_graph = subgraphs[0].copy()

#     # Intersect with each subsequent graph
#     for graph in subgraphs[1:]:
#         composed_graph = nx.compose(composed_graph, graph)

#     # Draw the subgraph
#     draw_subgraph(composed_graph, concept_id_to_concept, f"graphs/{' '.join(nodes)}")

#     assert composed_graph.number_of_nodes() > 1


# def test_find_semantic_and_adjacent_nodes_intersect(loaded_data):
#     G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     model = AutoModel.from_pretrained("bert-base-uncased")

#     nodes = ["Swollen ankle", "soft", "pains like needle", "bluish in color", "painful", "impact"]
#     subgraphs = []
#     for node in nodes:
#         # Generate embeddings
#         inputs = tokenizer(node, return_tensors="pt", padding=True, truncation=True).to("cpu")
#         outputs = model(**inputs)
#         embeddings = outputs.last_hidden_state.mean(dim=1).detach()

#         closest_nodes = find_closest_nodes(faiss_index, G, embeddings, cutoff_score=0.1)

#         if len(closest_nodes) > 0:
#             inward, outward, neighbors = find_adjacent_nodes([int(x[0]) for x in closest_nodes], G, n=1)
#             subgraphs.append(neighbors)
#             # draw_subgraph(neighbors, concept_id_to_concept, f"graphs/{' '.join(nodes)}")

#     subgraphs = [x for y in subgraphs for x in y if x.number_of_nodes() > 1]

#     if len(subgraphs) > 0:
#         intersection_graph = intersect_subgraphs(subgraphs)
#         if intersection_graph.number_of_edges() > 0:
#             draw_subgraph(intersection_graph, concept_id_to_concept, f"graphs/{' '.join(nodes)}")

#     assert intersection_graph.number_of_nodes() > 1


def test_find_related_terms(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    user_terms = ["Swollen ankle", "soft", "pains like needle", "bluish in color", "painful", "impact"]

    # Find related terms based on the user's query
    related_nodes = find_related_terms(user_terms, G, faiss_index, model, tokenizer)

    # Convert node IDs to their corresponding concepts for better readability
    related_concepts = [concept_id_to_concept.get(str(node), str(node)) for node in related_nodes]

    # Assert that we have found some related nodes
    assert len(related_nodes) > 0

    # Optional: Print or log the related nodes and concepts
    print("Related Nodes:", related_nodes)
    print("Related Concepts:", related_concepts)


# def test_find_closest_nodes(loaded_data):
#     G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

#     node = "pain in right wrist"

#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     model = AutoModel.from_pretrained("bert-base-uncased")

#     # Generate embeddings
#     inputs = tokenizer(node, return_tensors="pt", padding=True, truncation=True).to("cpu")
#     outputs = model(**inputs)
#     embeddings = outputs.last_hidden_state.mean(dim=1).detach()

#     closest_nodes = find_closest_nodes(faiss_index, G, embeddings, cutoff_score=0.1)
#     assert len(closest_nodes) == 2

#     assert isinstance(closest_nodes, list)
#     for node, score in closest_nodes:
#         assert isinstance(node, str)
#         assert isinstance(score, float)
#         assert [concept_id_to_concept[str(x[0])] for x in closest_nodes] == [
#             "Pain in right arm (finding)",
#             "Pain in right upper limb",
#         ]
