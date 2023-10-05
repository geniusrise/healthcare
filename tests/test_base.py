# import networkx as nx
# import pytest
# from transformers import AutoModel, AutoTokenizer

# from geniusrise_healthcare.base import (
#     find_symptoms_diseases,
#     generate_follow_up_questions_from_concepts,
#     generate_snomed_graph_from_concepts,
# )
# from geniusrise_healthcare.io import (
#     load_concept_dict,
#     load_faiss_index,
#     load_networkx_graph,
# )
# from geniusrise_healthcare.model import load_huggingface_model

# # Initialization settings
# LLM_MODEL = "/run/media/ixaxaar/models_q/CodeLlama-34B-Python-GPTQ"
# NER_MODEL = "bert-base-uncased"
# NETWORKX_GRAPH = "./saved/snomed.graph"
# FAISS_INDEX = "./saved/faiss.index"
# CONCEPT_ID_TO_CONCEPT = "./saved/concept_id_to_concept.pickle"
# DESCRIPTION_ID_TO_CONCEPT = "./saved/description_id_to_concept.pickle"

# # Load models and data
# model, tokenizer = load_huggingface_model(
#     model_name=LLM_MODEL,
#     use_cuda=True,
#     precision="float16",
#     quantize=False,
#     quantize_bits=8,
#     use_safetensors=True,
#     trust_remote_code=True,
# )

# G = load_networkx_graph(NETWORKX_GRAPH)
# faiss_index = load_faiss_index(FAISS_INDEX)
# concept_id_to_concept = load_concept_dict(CONCEPT_ID_TO_CONCEPT)
# description_id_to_concept = load_concept_dict(DESCRIPTION_ID_TO_CONCEPT)

# if NER_MODEL != "bert-base-uncased":
#     ner_model, ner_tokenizer = load_huggingface_model(
#         model_name=NER_MODEL,
#         use_cuda=True,
#         device_map=None,
#         precision="float32",
#         model_class_name="AutoModel",
#     )
# else:
#     ner_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     ner_model = AutoModel.from_pretrained("bert-base-uncased")

# # Test parameters
# test_user_inputs = ["I have a fever and cough."]


# @pytest.mark.parametrize("user_input", test_user_inputs)
# def test_find_symptoms_diseases(user_input, setup_data):
#     result = find_symptoms_diseases(
#         user_input=user_input,
#         tokenizer=tokenizer,
#         model=model,
#         ner_model=model,
#         ner_tokenizer=tokenizer,
#         type_ids_filter=[],
#         faiss_index=faiss_index,
#     )
#     assert "query" in result
#     assert "symptoms_diseases" in result
#     assert "snomed_concepts" in result


# @pytest.mark.parametrize("user_input", test_user_inputs)
# def test_generate_follow_up_questions_from_concepts(user_input, setup_data):
#     snomed_concepts = [[12345, 67890], [11111, 22222]]
#     result = generate_follow_up_questions_from_concepts(
#         snomed_concepts=snomed_concepts,
#         tokenizer=tokenizer,
#         model=model,
#         concept_id_to_concept=concept_id_to_concept,
#     )
#     assert isinstance(result, list)
#     for item in result:
#         assert "snomed_concepts" in item


# @pytest.mark.parametrize("user_input", test_user_inputs)
# def test_generate_snomed_graph_from_concepts(user_input, setup_data):
#     snomed_concepts = [[12345, 67890], [11111, 22222]]
#     graph, tmp_file, human_readable_str = generate_snomed_graph_from_concepts(
#         snomed_concepts=snomed_concepts,
#         G=G,
#         concept_id_to_concept=concept_id_to_concept,
#     )
#     assert isinstance(graph, nx.DiGraph)
#     assert tmp_file.endswith(".png")
#     assert isinstance(human_readable_str, str)
