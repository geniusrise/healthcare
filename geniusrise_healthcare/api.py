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

from fastapi import FastAPI, HTTPException, Query
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import logging

from geniusrise_healthcare.knowledge_graphs.base import load_graph
from geniusrise_healthcare.knowledge_graphs.network import NetworkxAPI
from geniusrise_healthcare.knowledge_graphs.index import IndexAPI
from geniusrise_healthcare.knowledge_graphs.vector import VectorAPI

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the graph
graph_name = "umls"  # You can change this to the specific graph you want to load
G = load_graph(graph_name)

# Initialize the APIs
networkx_api = NetworkxAPI(app, graph_name)
index_api = IndexAPI(app, graph_name, index_dir="./lucene_indexes")
vector_api = VectorAPI(app, graph_name, model_name="all-MiniLM-L6-v2", index_dir="./faiss_indexes")


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Medical Knowledge Graph API"}


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Graph information endpoint
@app.get("/graph_info")
async def graph_info():
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "graph_name": graph_name}


# Include the routes from the individual APIs
# NetworkxAPI routes
app.include_router(networkx_api.app.router)

# IndexAPI routes
app.include_router(index_api.app.router)

# VectorAPI routes
app.include_router(vector_api.app.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
