import pytest

from geniusrise_healthcare.io import (
    load_networkx_graph,
    load_faiss_index,
    load_concept_dict,
)
from geniusrise_healthcare.search import (
    find_related_subgraphs,
)
from geniusrise_healthcare.network_utils import (
    find_largest_strongly_connected_component,
    find_largest_weakly_connected_component,
    find_largest_attracting_component,
    find_largest_connected_component,
)
from geniusrise_healthcare.util import draw_subgraph
from transformers import AutoTokenizer, AutoModel


@pytest.fixture(scope="module")
def loaded_data():
    G = load_networkx_graph("./saved/snomed.graph")
    faiss_index = load_faiss_index("./saved/faiss.index.old")
    concept_id_to_concept = load_concept_dict("./saved/concept_id_to_concept.pickle")
    description_id_to_concept = load_concept_dict("./saved/description_id_to_concept.pickle")
    return G, faiss_index, concept_id_to_concept, description_id_to_concept


def test_find_largest_strongly_connected_component(loaded_data):
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
        semantic_types="disorder",
    )
    top_subgraph = find_largest_strongly_connected_component(top_subgraphs)

    draw_subgraph(top_subgraph, concept_id_to_concept, f"graphs/strongly_connected_component-{' '.join(user_terms)}")
    assert top_subgraph.number_of_nodes() > 0
    assert top_subgraph.number_of_edges() >= 0


def test_find_largest_weakly_connected_component(loaded_data):
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
        semantic_types="disorder",
    )
    top_subgraph = find_largest_weakly_connected_component(top_subgraphs)

    draw_subgraph(top_subgraph, concept_id_to_concept, f"graphs/weakly_connected_component-{' '.join(user_terms)}")
    assert top_subgraph.number_of_nodes() > 0
    assert top_subgraph.number_of_edges() > 0


def test_find_largest_attracting_component(loaded_data):
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
        semantic_types="disorder",
    )
    top_subgraph = find_largest_attracting_component(top_subgraphs)

    draw_subgraph(top_subgraph, concept_id_to_concept, f"graphs/attracting_component-{' '.join(user_terms)}")
    assert top_subgraph.number_of_nodes() > 0
    assert top_subgraph.number_of_edges() >= 0


def test_find_largest_connected_component(loaded_data):
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
        semantic_types="disorder",
    )
    top_subgraph = find_largest_connected_component(top_subgraphs)

    draw_subgraph(top_subgraph, concept_id_to_concept, f"graphs/connected_component-{' '.join(user_terms)}")
    assert top_subgraph.number_of_nodes() > 0
    assert top_subgraph.number_of_edges() > 0


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
        semantic_types="disorder",
        max_depth=5,
    )
    draw_subgraph(top_subgraphs, concept_id_to_concept, f"graphs/subgraphs-{' '.join(user_terms)}")
    assert len(top_subgraphs) > 2
