import networkx as nx
import pytest
from transformers import AutoModel, AutoTokenizer

from geniusrise_healthcare.io import load_concept_dict, load_faiss_index, load_networkx_graph
from geniusrise_healthcare.model import load_huggingface_model
from geniusrise_healthcare.search import find_adjacent_nodes, find_related_subgraphs, find_semantically_similar_nodes
from geniusrise_healthcare.util import draw_subgraph

# MODEL = "/run/media/ixaxaar/hynix_2tb/models/Llama-2-7b-hf"
# NETWORKX_GRAPH = "./saved-llama-7b/snomed.graph"
# FAISS_INDEX = "./saved-llama-7b/faiss.index"
# CONCEPT_ID_TO_CONCEPT = "./saved-llama-7b/concept_id_to_concept.pickle"
# DESCRIPTION_ID_TO_CONCEPT = "./saved-llama-7b/description_id_to_concept.pickle"


MODEL = "bert-base-uncased"
NETWORKX_GRAPH = "./saved/snomed.graph"
FAISS_INDEX = "./saved/faiss.index.old"
CONCEPT_ID_TO_CONCEPT = "./saved/concept_id_to_concept.pickle"
DESCRIPTION_ID_TO_CONCEPT = "./saved/description_id_to_concept.pickle"


@pytest.fixture(scope="module")
def loaded_data():
    G = load_networkx_graph(NETWORKX_GRAPH)
    faiss_index = load_faiss_index(FAISS_INDEX)
    concept_id_to_concept = load_concept_dict(CONCEPT_ID_TO_CONCEPT)
    description_id_to_concept = load_concept_dict(DESCRIPTION_ID_TO_CONCEPT)

    if MODEL != "bert-base-uncased":
        model, tokenizer = load_huggingface_model(
            MODEL, use_cuda=True, device_map=None, precision="float32", model_class_name="AutoModel"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
    return G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model


def test_find_semantic_and_adjacent_nodes_compose(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

    nodes = ["chest pain", "shortness of breath"]
    subgraphs = []
    for node in nodes:
        # Generate embeddings
        inputs = tokenizer(node, return_tensors="pt", padding=True, truncation=True).to("cpu")
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach()

        closest_nodes = find_semantically_similar_nodes(faiss_index, embeddings, cutoff_score=0.1)

        if len(closest_nodes) > 0:
            neighbors = find_adjacent_nodes([int(x[0]) for x in closest_nodes], G, n=1, top_n=5)
            subgraphs.append(neighbors)

    subgraphs = [x for y in subgraphs for x in y if x.number_of_nodes() > 1]
    composed_graph = subgraphs[0].copy()

    # Intersect with each subsequent graph
    for graph in subgraphs[1:]:
        composed_graph = nx.compose(composed_graph, graph)

    # Draw the subgraph
    draw_subgraph(composed_graph, concept_id_to_concept, f"graphs/adjacent-{' '.join(nodes)}")
    assert composed_graph.number_of_nodes() > len(nodes)


def test_find_related_subgraphs(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

    nodes = ["chest pain", "shortness of breath"]

    # Find related terms based on the user's query
    subgraph, _ = find_related_subgraphs(
        nodes,
        G,
        faiss_index,
        model,
        tokenizer,
        concept_id_to_concept,
        cutoff_score=0.1,
        semantic_types="disorder",
        max_depth=2,
    )

    draw_subgraph(subgraph, concept_id_to_concept, f"graphs/related-subgraphs-{' '.join(nodes)}")
    assert len(subgraph) > len(nodes)


def test_find_semantically_similar_nodes(loaded_data):
    G, faiss_index, concept_id_to_concept, description_id_to_concept, tokenizer, model = loaded_data

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
