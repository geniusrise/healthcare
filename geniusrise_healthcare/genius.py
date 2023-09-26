# import os
# from geniusrise import BatchInput, BatchOutput, Bolt, State, Spout
# from transformers import AutoConfig
# import tempfile
# from typing import Any, List, Optional
# import cherrypy
# import tempfile

# from .base import SnomedProcessor, SnomedTagger


# class SnomedAPI(Bolt):
#     networkx_path: str
#     faiss_path: str
#     type_id_dict_path: str
#     cat_model_path: str

#     def __init__(
#         self,
#         input: BatchInput,
#         output: BatchOutput,
#         state: State,
#         **kwargs,
#     ) -> None:
#         """
#         Initialize the bolt.

#         Args:
#             output (OutputConfig): The output data.
#             state (State): The state manager.
#             **kwargs: Additional keyword arguments.
#         """
#         super().__init__(
#             input=input,
#             output=output,
#             state=state,
#         )
#         self.input = input
#         self.output = output
#         self.state = state

#         self.st = SnomedTagger(
#             networkx_path=os.path.join(self.input.get(), self.networkx_path),
#             faiss_path=os.path.join(self.input.get(), self.faiss_path),
#             type_id_dict_path=os.path.join(self.input.get(), self.type_id_dict_path),
#         )

#     def _check_auth(self, username, password):
#         auth_header = cherrypy.request.headers.get("Authorization")
#         if auth_header:
#             auth_decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
#             provided_username, provided_password = auth_decoded.split(":", 1)
#             if provided_username != username or provided_password != password:
#                 raise cherrypy.HTTPError(401, "Unauthorized")
#         else:
#             raise cherrypy.HTTPError(401, "Unauthorized")

#     @cherrypy.expose
#     @cherrypy.tools.json_in()
#     def default(self, username=None, password=None):
#         uploaded_file = cherrypy.request.body.params["uploaded_file"].file
#         temp_file_path = os.path.join(tempfile.gettempdir(), "uploaded_document")
#         with open(temp_file_path, "wb") as temp_file:
#             while True:
#                 data = uploaded_file.read(8192)
#                 if not data:
#                     break
#                 temp_file.write(data)

#         data = cherrypy.request.json

#         return self.st.tag_and_find_nodes(
#             document_path=temp_file_path,
#             document_type=data["document_type"],
#             type_ids_filter=data["allowed_types"],
#             cutoff_score=data["cutoff_score"],
#             n_hops=data["adjacent_hops"],
#             cat_model_path=self.cat_model_path,
#         )

#     def listen(
#         self,
#         networkx_path: str,
#         faiss_path: str,
#         type_id_dict_path: str,
#         cat_model_path: str,
#         endpoint: str = "*",
#         port: int = 3000,
#         username: Optional[str] = None,
#         password: Optional[str] = None,
#     ):
#         """
#         Start listening for API calls.

#         Args:
#             endpoint (str): The API endpoint to listen to. Defaults to "*".
#             port (int): The port to listen on. Defaults to 3000.
#             username (Optional[str]): The username for basic authentication. Defaults to None.
#             password (Optional[str]): The password for basic authentication. Defaults to None.
#         """
#         self.networkx_path = networkx_path
#         self.faiss_path = faiss_path
#         self.type_id_dict_path = type_id_dict_path
#         self.cat_model_path = cat_model_path

#         cherrypy.config.update(
#             {
#                 "server.socket_host": "0.0.0.0",
#                 "server.socket_port": port,
#                 "log.screen": False,
#             }
#         )
#         cherrypy.tree.mount(self, "/")
#         cherrypy.engine.start()
#         cherrypy.engine.block()


# class Snomed(Bolt):
#     snomed_zip_path: str
#     networkx_save_path: str
#     faiss_save_path: str
#     type_id_dict_save_path: str
#     tokenizer_class: str
#     model_name: str
#     config: Any
#     model: Any
#     tokenizer: Any

#     def __init__(
#         self,
#         input: BatchInput,
#         output: BatchOutput,
#         state: State,
#         **kwargs,
#     ) -> None:
#         """
#         Initialize the bolt.

#         Args:
#             input (BatchInput): The batch input data.
#             output (OutputConfig): The output data.
#             state (State): The state manager.
#             **kwargs: Additional keyword arguments.
#         """
#         super().__init__(
#             input=input,
#             output=output,
#             state=state,
#         )
#         self.input = input
#         self.output = output
#         self.state = state

#     def process(
#         self,
#         snomed_zip_path="SnomedCT_InternationalRF2_PRODUCTION_20230901T120000Z.zip",
#         networkx_save_path: str = "networkx",
#         faiss_save_path: str = "faiss",
#         type_id_dict_save_path: str = "type_id_dict",
#         tokenizer_class: str = "AutoTokenizer",
#         model_name: str = "bert-base-uncased",
#         **kwargs,
#     ) -> None:
#         self.snomed_zip_path = snomed_zip_path
#         self.networkx_save_path = networkx_save_path
#         self.faiss_save_path = faiss_save_path
#         self.type_id_dict_save_path = type_id_dict_save_path
#         self.tokenizer_class = tokenizer_class
#         self.model_name = model_name

#         # load the tokenizer

#         self.config = AutoConfig.from_pretrained(self.model_name)
#         self.model = getattr(__import__("transformers"), str(self.model_class)).from_pretrained(
#             self.model_name, config=self.config
#         )
#         self.tokenizer = getattr(__import__("transformers"), str(self.tokenizer_class)).from_pretrained(self.model_name)

#         # Load, preprocess, and save SNOMED into NetworkX, FAISS and type_id_dict
#         st = SnomedProcessor()
#         st.load_and_save_snomed(
#             snomed_zip_path=snomed_zip_path,
#             snomed_extract_path=tempfile.mkdtemp(),
#             networkx_save_path=os.path.join(self.output.get(), networkx_save_path),
#             faiss_save_path=os.path.join(self.output.get(), faiss_save_path),
#             type_id_dict_save_path=os.path.join(self.output.get(), type_id_dict_save_path),
#             tokenizer=self.tokenizer,
#             model=self.model,
#         )
