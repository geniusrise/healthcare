import tempfile
from typing import Any, Dict, List, Tuple
import logging

import networkx as nx
import pandas as pd
from transformers import AutoTokenizer, GenerationMixin

import faiss  # type: ignore
from geniusrise_healthcare.model import generate_embeddings
from geniusrise_healthcare.ner import annotate_snomed
from geniusrise_healthcare.qa import generate_follow_up_questions
from geniusrise_healthcare.search import find_adjacent_nodes, find_semantically_similar_nodes
from geniusrise_healthcare.summary import generate_summary
from geniusrise_healthcare.util import draw_subgraph


log = logging.getLogger(__file__)


def find_symptoms_diseases(
    user_input: str,
    tokenizer: AutoTokenizer,
    model: GenerationMixin,
    ner_model: Any,
    ner_tokenizer: Any,
    concept_id_to_concept: Dict[str, str],
    type_ids_filter: List[str],
    faiss_index: faiss.IndexIDMap,  # type: ignore
    semantic_similarity_cutoff: float = 0.1,
) -> Dict[str, Any]:
    """
    Find symptoms and diseases based on user input and return related SNOMED concepts.

    Parameters:
    - user_input (str): The user's input text.
    - tokenizer (AutoTokenizer): The Hugging Face tokenizer instance.
    - model (GenerationMixin): The Hugging Face model instance.
    - ner_model (Any): The NER model.
    - ner_tokenizer (Any): The NER tokenizer.
    - type_ids_filter (List[str]): List of type IDs to filter annotations.
    - faiss_index (faiss.IndexIDMap): The FAISS index.
    - semantic_similarity_cutoff (float, optional): The similarity score below which nodes will be ignored.

    Returns:
    - Dict[str, Any]: A dictionary containing the query, symptoms, diseases, and related SNOMED concepts.
    """
    data = pd.DataFrame({"text": [user_input]})
    annotations = annotate_snomed("llm", tokenizer, model, data, type_ids_filter, max_new_tokens=25)
    symptoms_and_diseases = [x["snomed"] for x in annotations[0]["annotations"]]

    snomed_concept_ids = []
    for node in symptoms_and_diseases:
        embeddings = generate_embeddings(term=node, tokenizer=ner_tokenizer, model=ner_model)
        closest_nodes = find_semantically_similar_nodes(
            faiss_index=faiss_index, embedding=embeddings, cutoff_score=semantic_similarity_cutoff
        )
        if len(closest_nodes) > 0:
            snomed_concept_ids.append([int(x[0]) for x in closest_nodes])

    return {
        "query": user_input,
        "symptoms_diseases": symptoms_and_diseases,
        "snomed_concept_ids": snomed_concept_ids,
        "snomed_concepts": [[concept_id_to_concept[str(y)] for y in x] for x in snomed_concept_ids],
    }


def generate_follow_up_questions_from_concepts(
    snomed_concept_ids: List[List[int]],
    tokenizer: AutoTokenizer,
    model: GenerationMixin,
    concept_id_to_concept: Dict[str, str],
    decoding_strategy: str = "generate",
    **generation_params: Any,
) -> List[dict]:
    """
    Generate follow-up questions based on SNOMED concepts.

    Parameters:
    - snomed_concept_ids (List[List[int]]): List of SNOMED concept identifiers.
    - tokenizer (AutoTokenizer): The Hugging Face tokenizer instance.
    - model (GenerationMixin): The Hugging Face model instance.
    - concept_id_to_concept (Dict[str, str]): Mapping from concept IDs to concepts.
    - decoding_strategy (str, optional): The decoding strategy to use for text generation.
    - **generation_params (Any): Additional parameters for text generation.

    Returns:
    - List[dict]: A list of dictionaries containing the generated follow-up questions.
    """
    all_follow_up_questions = []
    for _conditions in snomed_concept_ids:
        conditions = [concept_id_to_concept.get(str(node), "0") for node in _conditions]
        follow_up_questions = generate_follow_up_questions(
            tokenizer=tokenizer,
            model=model,
            data=conditions,
            decoding_strategy=decoding_strategy,
            **generation_params,
        )
        all_follow_up_questions.append(
            {
                "snomed_concept_ids": _conditions,
                "snomed_concepts": conditions,
                "questions": follow_up_questions["follow_up_questions"],
            }
        )

    return all_follow_up_questions


def generate_summary_from_qa(
    snomed_concept_ids: List[List[int]],
    qa: Dict[str, str],
    tokenizer: AutoTokenizer,
    model: GenerationMixin,
    concept_id_to_concept: Dict[str, str],
    decoding_strategy: str = "generate",
    **generation_params: Any,
) -> Dict[str, Any]:
    """
    Generate follow-up questions based on SNOMED concepts.

    Parameters:
    - snomed_concept_ids (List[List[int]]): List of SNOMED concept identifiers.
    - tokenizer (AutoTokenizer): The Hugging Face tokenizer instance.
    - model (GenerationMixin): The Hugging Face model instance.
    - concept_id_to_concept (Dict[str, str]): Mapping from concept IDs to concepts.
    - decoding_strategy (str, optional): The decoding strategy to use for text generation.
    - **generation_params (Any): Additional parameters for text generation.

    Returns:
    - List[dict]: A list of dictionaries containing the generated follow-up questions.
    """

    snomed_concepts = [concept_id_to_concept[str(y)] for x in snomed_concept_ids for y in x]
    result = generate_summary(
        tokenizer=tokenizer,
        model=model,
        conditions=snomed_concepts,
        qa=qa,
        decoding_strategy=decoding_strategy,
        **generation_params,
    )

    return result


def generate_snomed_graph_from_concepts(
    snomed_concepts: List[List[int]],
    G: nx.DiGraph,
    concept_id_to_concept: Dict[str, str],
    graph_search_depth: int = 1,
) -> Tuple[List[nx.DiGraph], str, str]:
    """
    Generate SNOMED graphs based on SNOMED concepts.

    Parameters:
    - snomed_concepts (List[int]): List of SNOMED concepts.
    - G (nx.DiGraph): The NetworkX graph.
    - graph_search_depth (int, optional): Depth for graph search.

    Returns:
    - Tuple[List[nx.DiGraph], str, str]: A graph and a location of its image and a human-readable string.
    """
    subgraphs = []
    for node in snomed_concepts:
        all_neighbors = find_adjacent_nodes(source_nodes=node, G=G, n=graph_search_depth, top_n=0)
        subgraphs.extend(all_neighbors)

    composed_graph = subgraphs[0].copy()
    for graph in subgraphs[1:]:
        composed_graph = nx.compose(composed_graph, graph)

    tmp_file = f"{tempfile.mkdtemp()}/image"
    log.info(f"Saving image at {tmp_file}")

    draw_subgraph(
        subgraph=composed_graph,
        concept_id_to_concept=concept_id_to_concept,
        save_location=tmp_file,
        # highlight_nodes=[x for y in snomed_concepts for x in y],
    )

    human_readable_str = "Graph:\n"
    for edge in composed_graph.edges(data=True):
        from_node, to_node, edge_data = edge
        from_node_name = concept_id_to_concept.get(str(from_node), from_node)
        to_node_name = concept_id_to_concept.get(str(to_node), to_node)
        # edge_data_str = ", ".join(f"{k}={v}" for k, v in .items())
        human_readable_str += (
            f"{from_node_name} --[{ concept_id_to_concept[edge_data['relationship_type']] }]--> {to_node_name}\n"
        )

    return composed_graph, f"{tmp_file}.png", human_readable_str
