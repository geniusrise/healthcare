from typing import Any, Dict, List

import networkx as nx
import pandas as pd
from transformers import AutoTokenizer, GenerationMixin

import faiss  # type: ignore
from geniusrise_healthcare.model import generate_embeddings
from geniusrise_healthcare.ner import annotate_snomed
from geniusrise_healthcare.qa import generate_follow_up_questions
from geniusrise_healthcare.search import (
    find_adjacent_nodes,
    find_semantically_similar_nodes,
)
from geniusrise_healthcare.util import draw_subgraph


def generate_follow_up_questions_from_input(
    user_input: str,
    G: nx.DiGraph,
    faiss_index: faiss.IndexIDMap,  # type: ignore
    tokenizer: AutoTokenizer,
    model: GenerationMixin,
    ner_model: Any,
    ner_tokenizer: Any,
    concept_id_to_concept: dict,
    type_ids_filter: List[str],
    semantic_similarity_cutoff: float = 0.1,
    graph_search_depth: int = 1,
    graph_search_top_n: int = 6,
    max_depth: int = 3,
    decoding_strategy: str = "generate",
    **generation_params: Any,
) -> Dict[str, List[str]]:
    """
    Generate follow-up questions based on user input.

    Parameters:
    - user_input (str): The user's input text.
    - G (nx.DiGraph): The NetworkX graph.
    - faiss_index (faiss.IndexIDMap): The FAISS index.
    - tokenizer (AutoTokenizer): The Hugging Face tokenizer instance.
    - model (GenerationMixin): The Hugging Face model instance.
    - concept_id_to_concept (dict): Mapping from concept IDs to concepts.
    - type_ids_filter (List[str]): List of type IDs to filter annotations.
    - semantic_similarity_cutoff (float, optional): The similarity score below which nodes will be ignored.
    - max_depth (int, optional): Maximum depth for recursive search.
    - decoding_strategy (str, optional): The decoding strategy to use for text generation.
    - **generation_params (Any): Additional parameters for text generation.

    Returns:
    - Dict[str, List[str]]: A dictionary containing the generated follow-up questions.
    """

    # Step 1: Extract symptoms and diseases from user input
    data = pd.DataFrame({"text": [user_input]})
    annotations = annotate_snomed("llm", tokenizer, model, data, type_ids_filter)
    symptoms_and_diseases = [
        x["snomed"] for x in annotations[0]["annotations"]
    ]  # Assuming the first document contains the relevant data

    # Step 2: Search for semantically similar nodes in snomed
    topk_subgraphs: List[nx.DiGraph] = []
    subgraphs: List[nx.DiGraph] = []
    snomed_concepts = []
    for node in symptoms_and_diseases:
        # Generate embeddings
        embeddings = generate_embeddings(term=node, tokenizer=ner_tokenizer, model=ner_model)
        closest_nodes = find_semantically_similar_nodes(
            faiss_index=faiss_index,
            embedding=embeddings,
            cutoff_score=semantic_similarity_cutoff,
        )

        if len(closest_nodes) > 0:
            snomed_concepts.extend([int(x[0]) for x in closest_nodes])
            neighbors = find_adjacent_nodes(
                source_nodes=[int(x[0]) for x in closest_nodes],
                G=G,
                n=graph_search_depth,
                top_n=graph_search_top_n,
            )
            all_neighbors = find_adjacent_nodes(
                source_nodes=[int(x[0]) for x in closest_nodes],
                G=G,
                n=graph_search_depth,
                top_n=0,
            )
            topk_subgraphs.extend(neighbors)
            subgraphs.extend(all_neighbors)

    topk_subgraphs = [x for x in topk_subgraphs if x.number_of_nodes() > 1]

    # compose all neighbors into one graph to display the entire network
    composed_graph = subgraphs[0].copy()
    for graph in subgraphs[1:]:
        composed_graph = nx.compose(composed_graph, graph)
    draw_subgraph(
        subgraph=composed_graph,
        concept_id_to_concept=concept_id_to_concept,
        save_location=f"graphs/step2-{' '.join(symptoms_and_diseases)}",
        # highlight_nodes=snomed_concepts,
    )

    # Step 3: Generate follow-up questions for each subgraph
    all_follow_up_questions = []
    for subgraph in topk_subgraphs:
        related_nodes = list(subgraph.nodes())  # type: ignore
        conditions = [concept_id_to_concept.get(str(node), "0") for node in related_nodes]
        follow_up_questions = generate_follow_up_questions(
            tokenizer=tokenizer,
            model=model,
            data=conditions,
            decoding_strategy=decoding_strategy,
            **generation_params,
        )
        all_follow_up_questions.extend(follow_up_questions["follow_up_questions"])

    return {"follow_up_questions": all_follow_up_questions}


def generate_follow_up_questions_from_input_no_snomed(
    user_input: str,
    G: nx.DiGraph,
    faiss_index: faiss.IndexIDMap,  # type: ignore
    tokenizer: AutoTokenizer,
    model: GenerationMixin,
    ner_model: Any,
    ner_tokenizer: Any,
    concept_id_to_concept: dict,
    type_ids_filter: List[str],
    semantic_similarity_cutoff: float = 0.1,
    graph_search_depth: int = 1,
    graph_search_top_n: int = 6,
    max_depth: int = 3,
    decoding_strategy: str = "generate",
    **generation_params: Any,
) -> Dict[str, List[str]]:
    """
    Generate follow-up questions based on user input.

    Parameters:
    - user_input (str): The user's input text.
    - G (nx.DiGraph): The NetworkX graph.
    - faiss_index (faiss.IndexIDMap): The FAISS index.
    - tokenizer (AutoTokenizer): The Hugging Face tokenizer instance.
    - model (GenerationMixin): The Hugging Face model instance.
    - concept_id_to_concept (dict): Mapping from concept IDs to concepts.
    - type_ids_filter (List[str]): List of type IDs to filter annotations.
    - semantic_similarity_cutoff (float, optional): The similarity score below which nodes will be ignored.
    - max_depth (int, optional): Maximum depth for recursive search.
    - decoding_strategy (str, optional): The decoding strategy to use for text generation.
    - **generation_params (Any): Additional parameters for text generation.

    Returns:
    - Dict[str, List[str]]: A dictionary containing the generated follow-up questions.
    """

    # Step 1: Extract symptoms and diseases from user input
    data = pd.DataFrame({"text": [user_input]})
    annotations = annotate_snomed("llm", tokenizer, model, data, type_ids_filter)
    symptoms_and_diseases = [
        x["snomed"] for x in annotations[0]["annotations"]
    ]  # Assuming the first document contains the relevant data

    # Step 2: Search for semantically similar nodes in snomed
    subgraphs: List[nx.DiGraph] = []
    snomed_concepts = []
    for node in symptoms_and_diseases:
        # Generate embeddings
        embeddings = generate_embeddings(term=node, tokenizer=ner_tokenizer, model=ner_model)
        closest_nodes = find_semantically_similar_nodes(
            faiss_index=faiss_index,
            embedding=embeddings,
            cutoff_score=semantic_similarity_cutoff,
        )

        if len(closest_nodes) > 0:
            snomed_concepts.append([int(x[0]) for x in closest_nodes])
            all_neighbors = find_adjacent_nodes(
                source_nodes=[int(x[0]) for x in closest_nodes],
                G=G,
                n=graph_search_depth,
                top_n=0,
            )
            subgraphs.extend(all_neighbors)

    # compose all neighbors into one graph to display the entire network
    composed_graph = subgraphs[0].copy()
    for graph in subgraphs[1:]:
        composed_graph = nx.compose(composed_graph, graph)
    draw_subgraph(
        subgraph=composed_graph,
        concept_id_to_concept=concept_id_to_concept,
        save_location=f"graphs/step2-{' '.join(symptoms_and_diseases)}",
        highlight_nodes=snomed_concepts,
    )

    # Step 3: Generate follow-up questions for each subgraph
    all_follow_up_questions = []
    for _conditions in snomed_concepts:
        conditions = [concept_id_to_concept.get(str(node), "0") for node in _conditions]
        follow_up_questions = generate_follow_up_questions(
            tokenizer=tokenizer,
            model=model,
            data=conditions,
            decoding_strategy=decoding_strategy,
            **generation_params,
        )
        all_follow_up_questions.extend(follow_up_questions["follow_up_questions"])

    all_follow_up_questions = list(set(all_follow_up_questions))

    return {"follow_up_questions": all_follow_up_questions}


def follow_up(
    user_input: str,
    G: nx.DiGraph,
    faiss_index: faiss.IndexIDMap,  # type: ignore
    tokenizer: AutoTokenizer,
    model: GenerationMixin,
    ner_model: Any,
    ner_tokenizer: Any,
    concept_id_to_concept: dict,
    type_ids_filter: List[str],
    semantic_similarity_cutoff: float = 0.1,
    graph_search_depth: int = 1,
    graph_search_top_n: int = 6,
    max_depth: int = 3,
    decoding_strategy: str = "generate",
    **generation_params: Any,
) -> Dict[str, List[str]]:
    """
    Generate follow-up questions based on user input.

    Parameters:
    - user_input (str): The user's input text.
    - G (nx.DiGraph): The NetworkX graph.
    - faiss_index (faiss.IndexIDMap): The FAISS index.
    - tokenizer (AutoTokenizer): The Hugging Face tokenizer instance.
    - model (GenerationMixin): The Hugging Face model instance.
    - concept_id_to_concept (dict): Mapping from concept IDs to concepts.
    - type_ids_filter (List[str]): List of type IDs to filter annotations.
    - semantic_similarity_cutoff (float, optional): The similarity score below which nodes will be ignored.
    - max_depth (int, optional): Maximum depth for recursive search.
    - decoding_strategy (str, optional): The decoding strategy to use for text generation.
    - **generation_params (Any): Additional parameters for text generation.

    Returns:
    - Dict[str, List[str]]: A dictionary containing the generated follow-up questions.
    """

    # Step 1: Extract symptoms and diseases from user input
    data = pd.DataFrame({"text": [user_input]})
    annotations = annotate_snomed("llm", tokenizer, model, data, type_ids_filter)
    symptoms_and_diseases = [
        x["snomed"] for x in annotations[0]["annotations"]
    ]  # Assuming the first document contains the relevant data

    # Step 2: Search for semantically similar nodes in snomed
    snomed_concepts = []
    for node in symptoms_and_diseases:
        # Generate embeddings
        embeddings = generate_embeddings(term=node, tokenizer=ner_tokenizer, model=ner_model)
        closest_nodes = find_semantically_similar_nodes(
            faiss_index=faiss_index,
            embedding=embeddings,
            cutoff_score=semantic_similarity_cutoff,
        )

        if len(closest_nodes) > 0:
            snomed_concepts.append([int(x[0]) for x in closest_nodes])

    # Step 3: Generate follow-up questions for each subgraph
    all_follow_up_questions = []
    for _conditions in snomed_concepts:
        conditions = [concept_id_to_concept.get(str(node), "0") for node in _conditions]
        follow_up_questions = generate_follow_up_questions(
            tokenizer=tokenizer,
            model=model,
            data=conditions,
            decoding_strategy=decoding_strategy,
            **generation_params,
        )
        all_follow_up_questions.extend(follow_up_questions["follow_up_questions"])

    all_follow_up_questions = list(set(all_follow_up_questions))

    return {"follow_up_questions": all_follow_up_questions}
