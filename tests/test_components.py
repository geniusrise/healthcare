import pytest
from transformers import AutoModel, AutoTokenizer

from geniusrise_healthcare.io import (
    load_concept_dict,
    load_faiss_index,
    load_networkx_graph,
)
from geniusrise_healthcare.network_utils import (
    # find_largest_attracting_component,
    find_largest_connected_component,
    # find_largest_strongly_connected_component,
    # find_largest_weakly_connected_component,
    find_largest_connected_component_with_nodes,
)
from geniusrise_healthcare.search import find_related_subgraphs
from geniusrise_healthcare.util import draw_subgraph


QUERY = ["fever", "back pain", "shivering"]
SEMANTIC_TYPES = ["disorder"]


@pytest.fixture(scope="module")
def loaded_data():
    G = load_networkx_graph("./saved/snomed.graph")
    faiss_index = load_faiss_index("./saved/faiss.index.old")
    concept_id_to_concept = load_concept_dict("./saved/concept_id_to_concept.pickle")
    description_id_to_concept = load_concept_dict("./saved/description_id_to_concept.pickle")
    return G, faiss_index, concept_id_to_concept, description_id_to_concept


# def test_find_largest_strongly_connected_component(loaded_data):
#     G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     model = AutoModel.from_pretrained("bert-base-uncased")

#     user_terms = QUERY

#     # Find related terms based on the user's query
#     top_subgraphs, semantically_similar_nodes = find_related_subgraphs(
#         user_terms,
#         G,
#         faiss_index,
#         model,
#         tokenizer,
#         concept_id_to_concept,
#         cutoff_score=0.1,
#         semantic_types=SEMANTIC_TYPES,
#     )
#     top_subgraph = find_largest_strongly_connected_component(top_subgraphs)

#     draw_subgraph(
#         top_subgraph,
#         concept_id_to_concept,
#         f"graphs/strongly_connected_component-{' '.join(user_terms)}",
#         highlight_nodes=semantically_similar_nodes,
#     )
#     assert top_subgraph.number_of_nodes() > 0
#     assert top_subgraph.number_of_edges() >= 0


# def test_find_largest_weakly_connected_component(loaded_data):
#     G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     model = AutoModel.from_pretrained("bert-base-uncased")

#     user_terms = QUERY

#     # Find related terms based on the user's query
#     top_subgraphs, semantically_similar_nodes = find_related_subgraphs(
#         user_terms,
#         G,
#         faiss_index,
#         model,
#         tokenizer,
#         concept_id_to_concept,
#         cutoff_score=0.1,
#         semantic_types=SEMANTIC_TYPES,
#     )
#     top_subgraph = find_largest_weakly_connected_component(top_subgraphs)

#     draw_subgraph(
#         top_subgraph,
#         concept_id_to_concept,
#         f"graphs/weakly_connected_component-{' '.join(user_terms)}",
#         highlight_nodes=semantically_similar_nodes,
#     )
#     assert top_subgraph.number_of_nodes() > 0
#     assert top_subgraph.number_of_edges() > 0


# def test_find_largest_attracting_component(loaded_data):
#     G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     model = AutoModel.from_pretrained("bert-base-uncased")

#     user_terms = QUERY

#     # Find related terms based on the user's query
#     top_subgraphs, semantically_similar_nodes = find_related_subgraphs(
#         user_terms,
#         G,
#         faiss_index,
#         model,
#         tokenizer,
#         concept_id_to_concept,
#         cutoff_score=0.1,
#         semantic_types=SEMANTIC_TYPES,
#     )
#     top_subgraph = find_largest_attracting_component(top_subgraphs)

#     draw_subgraph(
#         top_subgraph,
#         concept_id_to_concept,
#         f"graphs/attracting_component-{' '.join(user_terms)}",
#         highlight_nodes=semantically_similar_nodes,
#     )
#     assert top_subgraph.number_of_nodes() > 0
#     assert top_subgraph.number_of_edges() >= 0


def test_find_largest_connected_component(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    user_terms = QUERY

    # Find related terms based on the user's query
    top_subgraphs, semantically_similar_nodes = find_related_subgraphs(
        user_terms,
        G,
        faiss_index,
        model,
        tokenizer,
        concept_id_to_concept,
        cutoff_score=0.1,
        semantic_types=SEMANTIC_TYPES,
        max_depth=2,
    )
    top_subgraph = find_largest_connected_component(top_subgraphs)

    draw_subgraph(
        top_subgraph,
        concept_id_to_concept,
        f"graphs/nodes_connected_component-{' '.join(user_terms)}",
        highlight_nodes=semantically_similar_nodes,
    )
    assert top_subgraph.number_of_nodes() > 0
    assert top_subgraph.number_of_edges() > 0


def test_find_largest_connected_component_with_nodes(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    user_terms = QUERY

    # Find related terms based on the user's query
    top_subgraphs, semantically_similar_nodes = find_related_subgraphs(
        user_terms,
        G,
        faiss_index,
        model,
        tokenizer,
        concept_id_to_concept,
        cutoff_score=0.1,
        semantic_types=SEMANTIC_TYPES,
        max_depth=2,
    )
    top_subgraph = find_largest_connected_component_with_nodes(top_subgraphs, semantically_similar_nodes)

    draw_subgraph(
        top_subgraph,
        concept_id_to_concept,
        f"graphs/connected_component-{' '.join(user_terms)}",
        highlight_nodes=semantically_similar_nodes,
    )
    assert top_subgraph.number_of_nodes() > 0
    assert top_subgraph.number_of_edges() > 0


def test_find_largest_connected_component_stop(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    user_terms = QUERY

    # Find related terms based on the user's query
    top_subgraphs, semantically_similar_nodes = find_related_subgraphs(
        user_terms,
        G,
        faiss_index,
        model,
        tokenizer,
        concept_id_to_concept,
        cutoff_score=0.1,
        semantic_types=SEMANTIC_TYPES,
        stop_at_semantic_types=SEMANTIC_TYPES,
        max_depth=2,
    )
    top_subgraph = find_largest_connected_component(top_subgraphs)

    draw_subgraph(
        top_subgraph,
        concept_id_to_concept,
        f"graphs/stop_connected_component-{' '.join(user_terms)}",
        highlight_nodes=semantically_similar_nodes,
    )
    assert top_subgraph.number_of_nodes() > 0
    assert top_subgraph.number_of_edges() > 0


def test_find_largest_connected_component_with_nodes_stop(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    user_terms = QUERY

    # Find related terms based on the user's query
    top_subgraphs, semantically_similar_nodes = find_related_subgraphs(
        user_terms,
        G,
        faiss_index,
        model,
        tokenizer,
        concept_id_to_concept,
        cutoff_score=0.1,
        semantic_types=SEMANTIC_TYPES,
        stop_at_semantic_types=SEMANTIC_TYPES,
        max_depth=2,
    )
    top_subgraph = find_largest_connected_component_with_nodes(top_subgraphs, semantically_similar_nodes)

    draw_subgraph(
        top_subgraph,
        concept_id_to_concept,
        f"graphs/stop_node_connected_component-{' '.join(user_terms)}",
        highlight_nodes=semantically_similar_nodes,
    )
    assert top_subgraph.number_of_nodes() > 0
    assert top_subgraph.number_of_edges() > 0


def test_find_related_subgraphs(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    user_terms = QUERY

    # Find related terms based on the user's query
    top_subgraphs, semantically_similar_nodes = find_related_subgraphs(
        user_terms,
        G,
        faiss_index,
        model,
        tokenizer,
        concept_id_to_concept,
        cutoff_score=0.1,
        semantic_types=SEMANTIC_TYPES,
        max_depth=2,
    )
    draw_subgraph(
        top_subgraphs,
        concept_id_to_concept,
        f"graphs/subgraphs-{' '.join(user_terms)}",
        highlight_nodes=semantically_similar_nodes,
    )
    assert len(top_subgraphs) > 2


def test_find_related_subgraphs_stop(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept = loaded_data

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    user_terms = QUERY

    # Find related terms based on the user's query
    top_subgraphs, semantically_similar_nodes = find_related_subgraphs(
        user_terms,
        G,
        faiss_index,
        model,
        tokenizer,
        concept_id_to_concept,
        cutoff_score=0.1,
        semantic_types=SEMANTIC_TYPES,
        stop_at_semantic_types=SEMANTIC_TYPES,
        max_depth=2,
    )
    draw_subgraph(
        top_subgraphs,
        concept_id_to_concept,
        f"graphs/stop_subgraphs-{' '.join(user_terms)}",
        highlight_nodes=semantically_similar_nodes,
    )
    assert len(top_subgraphs) > 2
