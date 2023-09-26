# import logging
# from typing import Any, Dict, List

# import docx
# import textract
# from medcat.cat import CAT
# from odf import teletype
# from odf import text as odf_text
# from PyPDF2 import PdfFileReader

# import pandas as pd
# import faiss

# from .load import (
#     load_snomed_into_networkx,
#     save_faiss_index,
#     save_networkx_graph,
#     save_concept_dict,
#     unzip_snomed_ct,
# )
# from .ner import annotate_text_with_snomed
# from .search import (
#     find_closest_nodes,
#     find_semantic_and_adjacent_nodes,
#     load_faiss_index,
#     load_networkx_graph,
#     load_concept_dict,
# )


# class SnomedTagger:
#     cat: CAT

#     def __init__(self, networkx_path: str, faiss_path: str, type_id_dict_path: str):
#         """
#         Initialize the SnomedTagger class.

#         Parameters:
#         - networkx_path (str): Path to the saved NetworkX graph.
#         - faiss_path (str): Path to the saved FAISS index.
#         - type_id_dict_path (str): Path to the saved description_id_to_concept dictionary.
#         - cat (CAT): The MedCAT Clinical Annotation Tool instance.
#         """
#         self.G = load_networkx_graph(networkx_path)
#         self.faiss_index = load_faiss_index(faiss_path)
#         self.description_id_to_concept = load_concept_dict(type_id_dict_path)

#     def tag_and_find_nodes(
#         self,
#         document_path: str,
#         document_type: str,
#         type_ids_filter: List[str],
#         cutoff_score: float,
#         n_hops: int,
#         cat_model_path: str,
#     ) -> Dict[str, Any]:
#         """
#         Tags the document and finds the tag nodes, semantic and adjacent nodes from SNOMED.

#         Parameters:
#         - document_path (str): Path to the document.
#         - document_type (str): Type of the document (pdf, txt, doc, docx, odf).
#         - type_ids_filter (List[str]): List of type IDs to filter annotations.
#         - cutoff_score (float): The similarity score below which nodes will be ignored.
#         - n_hops (int): The number of hops to consider for adjacency.
#         - cat_model_path (str): Path to the MedCAT model pack

#         Returns:
#         Dict[str, Any]: Result of the tagging and node finding process.
#         """
#         # Load CAT
#         self.cat = CAT.load_model_pack(cat_model_path)

#         # Read the document based on its type
#         if document_type == "pdf":
#             with open(document_path, "rb") as f:
#                 reader = PdfFileReader(f)
#                 text = " ".join([reader.getPage(i).extractText() for i in range(reader.getNumPages())])
#         elif document_type == "txt":
#             with open(document_path, "r") as f:
#                 text = f.read()
#         elif document_type == "doc":
#             text = textract.process(document_path, method="pythoncom_wv2").decode()
#         elif document_type == "docx":
#             doc = docx.Document(document_path)
#             text = " ".join([p.text for p in doc.paragraphs])
#         elif document_type == "odf":
#             textdoc = odf_text.TextDocument(document_path)
#             text = teletype.extractText(textdoc.body)
#         else:
#             logging.error(f"Unsupported document type: {document_type}")
#             return {"error": "Unsupported document type"}

#         # Create a DataFrame for the document
#         df = pd.DataFrame({"text": [text]})

#         # Annotate the text with SNOMED IDs
#         annotations = annotate_text_with_snomed(self.cat, df, type_ids_filter)

#         # Initialize result dictionary
#         result: dict = {"annotations": annotations}

#         # Find semantic and adjacent nodes for each annotation
#         for doc_id, doc_data in annotations.items():
#             for annotation in doc_data["annotations"]:
#                 cui: str = annotation["cui"]  # type: ignore
#                 closest_nodes = find_closest_nodes(self.faiss_index, self.G, cui, cutoff_score)
#                 semantic_and_adjacent, semantic_and_not_adjacent = find_semantic_and_adjacent_nodes(
#                     cui, [node[0] for node in closest_nodes], self.G, n_hops
#                 )

#                 # Add to result
#                 result.setdefault("semantic_and_adjacent", []).append(semantic_and_adjacent)
#                 result.setdefault("semantic_and_not_adjacent", []).append(semantic_and_not_adjacent)

#         return result


# class SnomedProcessor:
#     def load_and_save_snomed(
#         self,
#         snomed_zip_path: str,
#         snomed_extract_path: str,
#         networkx_save_path: str,
#         faiss_save_path: str,
#         type_id_dict_save_path: str,
#         tokenizer: Any,
#         model,
#     ) -> None:
#         """
#         Load SNOMED into NetworkX and FAISS, and save all three (NetworkX, FAISS, and type_id_dict) to given locations.

#         Parameters:
#         - snomed_zip_path (str): Path to the SNOMED zip file.
#         - snomed_extract_path (str): Directory where SNOMED will be extracted.
#         - networkx_save_path (str): Path where the NetworkX graph will be saved.
#         - faiss_save_path (str): Path where the FAISS index will be saved.
#         - type_id_dict_save_path (str): Path where the description_id_to_concept dictionary will be saved.
#         - tokenizer: Tokenizer for text embeddings.
#         - model: Model for generating text embeddings.

#         Returns:
#         None
#         """
#         # Unzip SNOMED
#         unzip_snomed_ct(snomed_zip_path, snomed_extract_path)

#         # Create a new FAISS index
#         faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(768))

#         # Load SNOMED into NetworkX and FAISS
#         self.G, self.description_id_to_concept = load_snomed_into_networkx(snomed_extract_path, tokenizer, model, faiss_index)

#         # Save NetworkX, FAISS, and type_id_dict
#         save_networkx_graph(self.G, networkx_save_path)
#         save_faiss_index(faiss_index, faiss_save_path)
#         save_concept_dict(self.description_id_to_concept, type_id_dict_save_path)
