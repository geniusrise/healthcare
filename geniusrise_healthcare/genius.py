import base64
from typing import Any, Dict, List, Optional
import os

import cherrypy
from geniusrise import BatchInput, BatchOutput, Bolt, State
from transformers import AutoModel, AutoTokenizer

from geniusrise_healthcare.base import (
    find_symptoms_diseases,
    generate_follow_up_questions_from_concepts,
    generate_snomed_graph_from_concepts,
    generate_summary_from_qa,
)
from geniusrise_healthcare.io import load_concept_dict, load_faiss_index, load_networkx_graph
from geniusrise_healthcare.model import load_huggingface_model


class InPatientAPI(Bolt):
    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ) -> None:
        super().__init__(input=input, output=output, state=state)

    def load_models(
        self,
        llm_model: str = "/run/media/ixaxaar/models_f/models/Mistral-7B-v0.1",
        ner_model: str = "bert-base-uncased",
        networkx_graph: str = "./saved/snomed.graph",
        faiss_index: str = "./saved/faiss.index",
        concept_id_to_concept: str = "./saved/concept_id_to_concept.pickle",
        description_id_to_concept: str = "./saved/description_id_to_concept.pickle",
    ) -> None:
        """Load all required models and tokenizers."""
        self.model, self.tokenizer = load_huggingface_model(
            llm_model,
            use_cuda=True,
            precision="float16",
            quantize=False,
            quantize_bits=8,
            use_safetensors=True,
            trust_remote_code=True,
        )
        self.G = load_networkx_graph(networkx_graph)
        self.faiss_index = load_faiss_index(faiss_index)
        self.concept_id_to_concept = load_concept_dict(concept_id_to_concept)
        self.description_id_to_concept = load_concept_dict(description_id_to_concept)

        if ner_model != "bert-base-uncased":
            self.ner_model, self.ner_tokenizer = load_huggingface_model(
                ner_model, use_cuda=True, device_map=None, precision="float32", model_class_name="AutoModel"
            )
        else:
            self.ner_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.ner_model = AutoModel.from_pretrained("bert-base-uncased")

    def _check_auth(self, username: str, password: str) -> None:
        """Check if the provided username and password are correct."""
        auth_header = cherrypy.request.headers.get("Authorization")
        if auth_header:
            auth_decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            provided_username, provided_password = auth_decoded.split(":", 1)
            if provided_username != username or provided_password != password:
                raise cherrypy.HTTPError(401, "Unauthorized")
        else:
            raise cherrypy.HTTPError(401, "Unauthorized")

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def find_symptoms_diseases(self, username: Optional[str] = None, password: Optional[str] = None) -> Dict[str, Any]:
        if username and password:
            self._check_auth(username=username, password=password)
        data = cherrypy.request.json
        user_input = data.get("user_input", "")
        type_ids_filter = data.get("type_ids_filter", [])
        semantic_similarity_cutoff = data.get("semantic_similarity_cutoff", 0.1)
        return find_symptoms_diseases(
            user_input=user_input,
            tokenizer=self.tokenizer,
            model=self.model,
            ner_model=self.ner_model,
            ner_tokenizer=self.ner_tokenizer,
            concept_id_to_concept=self.concept_id_to_concept,
            type_ids_filter=type_ids_filter,
            faiss_index=self.faiss_index,
            semantic_similarity_cutoff=semantic_similarity_cutoff,
        )

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def generate_follow_up_questions_from_concepts(
        self, username: Optional[str] = None, password: Optional[str] = None
    ) -> List[Dict]:
        if username and password:
            self._check_auth(username=username, password=password)
        data = cherrypy.request.json
        snomed_concept_ids = data.get("snomed_concept_ids", [])
        decoding_strategy = data.get("decoding_strategy", "generate")
        generation_params = data.get(
            "generation_params", {"temperature": 0.7, "do_sample": True, "max_new_tokens": 100}
        )
        return generate_follow_up_questions_from_concepts(
            snomed_concept_ids=snomed_concept_ids,
            tokenizer=self.tokenizer,
            model=self.model,
            concept_id_to_concept=self.concept_id_to_concept,
            decoding_strategy=decoding_strategy,
            **generation_params,
        )

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def generate_summary_from_qa(
        self, username: Optional[str] = None, password: Optional[str] = None
    ) -> Dict[str, Any]:
        if username and password:
            self._check_auth(username=username, password=password)
        data = cherrypy.request.json
        snomed_concept_ids = data.get("snomed_concept_ids", [])
        qa = data.get("qa", {})
        decoding_strategy = data.get("decoding_strategy", "generate")
        generation_params = data.get(
            "generation_params", {"temperature": 0.7, "do_sample": True, "max_new_tokens": 250}
        )
        return generate_summary_from_qa(
            snomed_concept_ids=snomed_concept_ids,
            qa=qa,
            tokenizer=self.tokenizer,
            model=self.model,
            concept_id_to_concept=self.concept_id_to_concept,
            decoding_strategy=decoding_strategy,
            **generation_params,
        )

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def generate_snomed_graph(self, username: Optional[str] = None, password: Optional[str] = None) -> None:
        if username and password:
            self._check_auth(username=username, password=password)
        data = cherrypy.request.json
        snomed_concepts = data.get("snomed_concepts", [])
        graph_search_depth = data.get("graph_search_depth", 1)
        result = generate_snomed_graph_from_concepts(
            snomed_concepts=snomed_concepts,
            G=self.G,
            concept_id_to_concept=self.concept_id_to_concept,
            graph_search_depth=graph_search_depth,
        )

        file_path = result[1]

        if os.path.exists(file_path):
            cherrypy.response.headers["Content-Type"] = "application/octet-stream"
            cherrypy.response.headers["Content-Disposition"] = f'attachment; filename="{os.path.basename(file_path)}"'

            with open(file_path, "rb") as f:
                file_data = f.read()

            return file_data  # type: ignore
        else:
            raise cherrypy.HTTPError(404, "File not found")

    def listen(
        self,
        endpoint: str = "*",
        port: int = 3000,
        llm_model: str = "/run/media/ixaxaar/models_f/models/Mistral-7B-v0.1",
        ner_model: str = "bert-base-uncased",
        networkx_graph: str = "./saved/snomed.graph",
        faiss_index: str = "./saved/faiss.index",
        concept_id_to_concept: str = "./saved/concept_id_to_concept.pickle",
        description_id_to_concept: str = "./saved/description_id_to_concept.pickle",
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.load_models(
            llm_model=llm_model,
            ner_model=ner_model,
            networkx_graph=networkx_graph,
            faiss_index=faiss_index,
            concept_id_to_concept=concept_id_to_concept,
            description_id_to_concept=description_id_to_concept,
        )
        cherrypy.config.update(
            {
                "server.socket_host": "0.0.0.0",
                "server.socket_port": port,
                "log.screen": False,
            }
        )
        cherrypy.tree.mount(self, "/")
        cherrypy.engine.start()
        cherrypy.engine.block()
