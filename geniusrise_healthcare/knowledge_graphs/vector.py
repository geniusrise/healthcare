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
import networkx as nx
from fastapi import FastAPI, HTTPException
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

from base import load_graph as load

log = logging.getLogger(__name__)


class VectorAPI:
    def __init__(
        self, app: FastAPI, graph_name: str, model_name: str = "all-MiniLM-L6-v2", index_dir: str = "./faiss_indexes"
    ):
        self.app = app
        self.graph_name = graph_name
        self.index_dir = index_dir
        self.model = SentenceTransformer(model_name)

        self.G = load(graph_name)
        self.index, self.reverse_map = self.load_or_create_faiss_index()

        self.setup_routes()

    def load_or_create_faiss_index(self):
        index_path = os.path.join(self.index_dir, f"{self.graph_name}_index.faiss")
        map_path = os.path.join(self.index_dir, f"{self.graph_name}_reverse_map.pkl")

        if os.path.exists(index_path) and os.path.exists(map_path):
            index = faiss.read_index(index_path)
            with open(map_path, "rb") as f:
                reverse_map = pickle.load(f)
        else:
            index, reverse_map = self.create_index()

            os.makedirs(self.index_dir, exist_ok=True)
            faiss.write_index(index, index_path)
            with open(map_path, "wb") as f:
                pickle.dump(reverse_map, f)

        return index, reverse_map

    def create_index(self):
        texts = []
        reverse_map = []

        for node, data in self.G.nodes(data=True):
            for key, value in data.items():
                texts.append(f"{key}: {value}")
                reverse_map.append(("node", node, key))

        for u, v, data in self.G.edges(data=True):
            for key, value in data.items():
                texts.append(f"{key}: {value}")
                reverse_map.append(("edge", (u, v), key))

        embeddings = self.model.encode(texts)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        return index, reverse_map

    def setup_routes(self):
        @self.app.get("/semantic_search")
        async def semantic_search(query: str, k: int = 10):
            query_embedding = self.model.encode([query])[0]
            D, I = self.index.search(query_embedding.reshape(1, -1), k)

            results = []
            for i, (distance, idx) in enumerate(zip(D[0], I[0])):
                item_type, item_id, field = self.reverse_map[idx]
                if item_type == "node":
                    data = self.G.nodes[item_id]
                else:  # edge
                    data = self.G.edges[item_id]
                results.append(
                    {
                        "type": item_type,
                        "id": item_id,
                        "field": field,
                        "value": data[field],
                        "similarity": float(distance),
                    }
                )
            return results

        @self.app.get("/semantic_relation_search")
        async def semantic_relation_search(source_query: str, relation_query: str, target_query: str, k: int = 10):
            source_embedding = self.model.encode([source_query])[0]
            relation_embedding = self.model.encode([relation_query])[0]
            target_embedding = self.model.encode([target_query])[0]

            _, source_I = self.index.search(source_embedding.reshape(1, -1), k)
            _, relation_I = self.index.search(relation_embedding.reshape(1, -1), k)
            _, target_I = self.index.search(target_embedding.reshape(1, -1), k)

            results = []
            for s_idx in source_I[0]:
                s_type, s_id, _ = self.reverse_map[s_idx]
                if s_type != "node":
                    continue

                for t_idx in target_I[0]:
                    t_type, t_id, _ = self.reverse_map[t_idx]
                    if t_type != "node":
                        continue

                    for r_idx in relation_I[0]:
                        r_type, r_id, r_field = self.reverse_map[r_idx]
                        if r_type != "edge":
                            continue

                        if r_id == (s_id, t_id):
                            results.append(
                                {
                                    "source": s_id,
                                    "target": t_id,
                                    "relation_field": r_field,
                                    "relation_value": self.G.edges[r_id][r_field],
                                    "source_data": dict(self.G.nodes[s_id]),
                                    "target_data": dict(self.G.nodes[t_id]),
                                    "edge_data": dict(self.G.edges[r_id]),
                                }
                            )
                            if len(results) >= k:
                                return results

            return results
