import pytest
from transformers import AutoModel, AutoTokenizer

from geniusrise_healthcare.io import load_concept_dict, load_faiss_index, load_networkx_graph
from geniusrise_healthcare.model import load_huggingface_model
from geniusrise_healthcare.network_utils import (  # find_largest_attracting_component,; find_largest_strongly_connected_component,; find_largest_weakly_connected_component,
    find_largest_connected_component,
    find_largest_connected_component_with_nodes,
)
from geniusrise_healthcare.search import find_related_subgraphs
from geniusrise_healthcare.util import draw_subgraph

QUERY = ["high fever", "back pain", "shivering"]
SEMANTIC_TYPES = ["disorder"]

# MODEL = "/run/media/ixaxaar/hynix_2tb/models/Llama-2-7b-hf"
# NETWORKX_GRAPH = "./saved-llama-7b/snomed.graph"
# FAISS_INDEX = "./saved-llama-7b/faiss.index"
# CONCEPT_ID_TO_CONCEPT = "./saved-llama-7b/concept_id_to_concept.pickle"
# DESCRIPTION_ID_TO_CONCEPT = "./saved-llama-7b/description_id_to_concept.pickle"


MODEL = "emilyalsentzer/Bio_ClinicalBERT"
NETWORKX_GRAPH = "./saved/snomed.graph"
FAISS_INDEX = "./saved/faiss.index"
CONCEPT_ID_TO_CONCEPT = "./saved/concept_id_to_concept.pickle"
DESCRIPTION_ID_TO_CONCEPT = "./saved/description_id_to_concept.pickle"


@pytest.fixture(scope="module")
def loaded_data():
    G = load_networkx_graph(NETWORKX_GRAPH)
    faiss_index = load_faiss_index(FAISS_INDEX)
    concept_id_to_concept = load_concept_dict(CONCEPT_ID_TO_CONCEPT)
    description_id_to_concept = load_concept_dict(DESCRIPTION_ID_TO_CONCEPT)

    if MODEL != "emilyalsentzer/Bio_ClinicalBERT":
        model, tokenizer = load_huggingface_model(
            MODEL, use_cuda=True, device_map=None, precision="float32", model_class_name="AutoModel"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model


def test_find_related_subgraphs(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

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
    G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

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


# def test_find_largest_strongly_connected_component(loaded_data):
#     G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

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
#     G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

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
#     G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

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
    G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

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
    G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

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
    G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

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
    G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

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
