# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import List, Dict, Any, Union
import networkx as nx
from fastapi import FastAPI, HTTPException
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)


class VectorAPI:
    def __init__(self, app: FastAPI, model_name: str = "all-MiniLM-L6-v2"):
        self.app = app
        self.graphs: Dict[str, nx.DiGraph] = {}
        self.embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.model = SentenceTransformer(model_name)
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/load_graph_and_compute_embeddings/{graph_name}")
        async def load_graph_and_compute_embeddings(graph_name: str):
            try:
                self.graphs[graph_name] = await self.load_graph(graph_name)
                await self.compute_embeddings(graph_name)
                return {"message": f"Graph {graph_name} loaded and embeddings computed successfully"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/semantic_search/{graph_name}")
        async def semantic_search(graph_name: str, query: str, k: int = 10):
            if graph_name not in self.graphs or graph_name not in self.embeddings:
                raise HTTPException(status_code=404, detail=f"Graph {graph_name} not loaded or embeddings not computed")

            query_embedding = self.model.encode([query])[0]
            results = self.search_embeddings(graph_name, query_embedding, k)
            return results

        @self.app.get("/semantic_relation_search/{graph_name}")
        async def semantic_relation_search(
            graph_name: str, source_query: str, relation_query: str, target_query: str, k: int = 10
        ):
            if graph_name not in self.graphs or graph_name not in self.embeddings:
                raise HTTPException(status_code=404, detail=f"Graph {graph_name} not loaded or embeddings not computed")

            source_embedding = self.model.encode([source_query])[0]
            relation_embedding = self.model.encode([relation_query])[0]
            target_embedding = self.model.encode([target_query])[0]

            results = self.search_relations(graph_name, source_embedding, relation_embedding, target_embedding, k)
            return results

    async def load_graph(self, graph_name: str) -> nx.DiGraph:
        # Implement graph loading logic here
        # This is a placeholder and should be replaced with actual graph loading code
        return nx.DiGraph()

    async def compute_embeddings(self, graph_name: str):
        G = self.graphs[graph_name]
        self.embeddings[graph_name] = {}

        # Compute node embeddings
        node_texts = [self.node_to_text(node, data) for node, data in G.nodes(data=True)]
        node_embeddings = self.model.encode(node_texts)
        self.embeddings[graph_name]["nodes"] = {node: emb for node, emb in zip(G.nodes(), node_embeddings)}

        # Compute edge embeddings
        edge_texts = [self.edge_to_text(u, v, data) for u, v, data in G.edges(data=True)]
        edge_embeddings = self.model.encode(edge_texts)
        self.embeddings[graph_name]["edges"] = {(u, v): emb for (u, v), emb in zip(G.edges(), edge_embeddings)}

    def node_to_text(self, node: str, data: Dict[str, Any]) -> str:
        # Convert node data to a string representation
        # This is a simple example and should be adapted based on your graph structure
        return f"Node {node}: {' '.join([f'{k}:{v}' for k, v in data.items()])}"

    def edge_to_text(self, u: str, v: str, data: Dict[str, Any]) -> str:
        # Convert edge data to a string representation
        # This is a simple example and should be adapted based on your graph structure
        return f"Edge from {u} to {v}: {' '.join([f'{k}:{v}' for k, v in data.items()])}"

    def search_embeddings(
        self, graph_name: str, query_embedding: np.ndarray, k: int
    ) -> List[Dict[str, Union[str, float]]]:
        node_embeddings = self.embeddings[graph_name]["nodes"]
        similarities = cosine_similarity([query_embedding], list(node_embeddings.values()))[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            node = list(node_embeddings.keys())[idx]
            results.append(
                {
                    "node": node,
                    "similarity": float(similarities[idx]),
                    "data": dict(self.graphs[graph_name].nodes[node]),
                }
            )
        return results

    def search_relations(
        self,
        graph_name: str,
        source_embedding: np.ndarray,
        relation_embedding: np.ndarray,
        target_embedding: np.ndarray,
        k: int,
    ) -> List[Dict[str, Union[str, float]]]:
        G = self.graphs[graph_name]
        edge_embeddings = self.embeddings[graph_name]["edges"]

        results = []
        for (u, v), edge_emb in edge_embeddings.items():
            source_sim = cosine_similarity([source_embedding], [self.embeddings[graph_name]["nodes"][u]])[0][0]
            target_sim = cosine_similarity([target_embedding], [self.embeddings[graph_name]["nodes"][v]])[0][0]
            relation_sim = cosine_similarity([relation_embedding], [edge_emb])[0][0]

            total_sim = (source_sim + relation_sim + target_sim) / 3
            results.append(
                {
                    "source": u,
                    "target": v,
                    "similarity": float(total_sim),
                    "edge_data": dict(G.edges[u, v]),
                    "source_data": dict(G.nodes[u]),
                    "target_data": dict(G.nodes[v]),
                }
            )

        return sorted(results, key=lambda x: x["similarity"], reverse=True)[:k]
