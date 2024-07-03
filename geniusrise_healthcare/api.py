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

import argparse
import uvicorn
from fastapi import FastAPI
import logging

from geniusrise_healthcare.knowledge_graphs import NetworkxAPI, IndexAPI, VectorAPI, load_graph


def create_app(graph_name: str, lucene_index_dir: str, faiss_index_dir: str, vector_model_name: str) -> FastAPI:
    app = FastAPI()

    # Load the graph
    G = load_graph(graph_name)

    # Initialize the APIs
    NetworkxAPI(app, G=G, graph_name=graph_name)
    IndexAPI(app, G=G, graph_name=graph_name, index_dir=lucene_index_dir)
    VectorAPI(app, G=G, graph_name=graph_name, model_name=vector_model_name, index_dir=faiss_index_dir)

    @app.get("/")
    async def root():
        return {"message": "Welcome to the Medical Knowledge Graph API"}

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.get("/graph_info")
    async def graph_info():
        return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "graph_name": graph_name}

    return app


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Start the FastAPI Knowledge Graph Server")
    parser.add_argument("--graph", type=str, default="umls", help="Name of the graph to load")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--lucene-index", type=str, default="./lucene_indexes", help="Directory for Lucene indexes")
    parser.add_argument("--faiss-index", type=str, default="./faiss_indexes", help="Directory for FAISS indexes")
    parser.add_argument("--vector-model", type=str, default="all-MiniLM-L6-v2", help="Name of the vector model to use")
    parser.add_argument( "--log-level", type=str, default="info", choices=["debug", "info", "warning", "error", "critical"], help="Logging level")
    # fmt: on

    args = parser.parse_args()

    # Configure logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=numeric_level)

    # Create the FastAPI app
    app = create_app(
        graph_name=args.graph,
        lucene_index_dir=args.lucene_index,
        faiss_index_dir=args.faiss_index,
        vector_model_name=args.vector_model,
    )

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
