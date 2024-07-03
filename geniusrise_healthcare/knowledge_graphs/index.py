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

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser

import networkx as nx
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any, Optional
from .base import load_graph as load
import tempfile


class IndexAPI:
    def __init__(
        self,
        app,
        G: nx.DiGraph,
        graph_name: str,
        index_dir: Optional[str],
    ):
        self.app = app
        self.index_dir_path = index_dir or tempfile.mkdtemp()

        self.G = G
        self.graph_name = graph_name

        # Initialize Lucene
        lucene.initVM(vmargs=["-Djava.awt.headless=true"])
        self.analyzer = StandardAnalyzer()
        self.index_dir = SimpleFSDirectory(Paths.get(self.index_dir_path))

        if not self.index_exists():
            self.index_graph(self.G)
        self.setup_routes()

    def index_exists(self) -> bool:
        """Check if the index already exists and is not empty."""
        try:
            reader = DirectoryReader.open(self.index_dir)
            exists = reader.numDocs() > 0
            reader.close()
            return exists
        except Exception:
            return False

    def setup_routes(self):
        @self.app.get("/index/search")
        async def search(query: str, limit: int = 10):
            try:
                results = self.search(query, limit)
                return results
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/index/get_node/{node_id}")
        async def get_node(node_id: str):
            try:
                node = self.get_node(node_id)
                return node
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))

        @self.app.get("/index/get_relation/{relation_id}")
        async def get_relation(relation_id: str):
            try:
                relation = self.get_relation(relation_id)
                return relation
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))

    def index_graph(self, G: nx.Graph):
        config = IndexWriterConfig(self.analyzer)
        writer = IndexWriter(self.index_dir, config)

        # Index nodes
        for node, data in G.nodes(data=True):
            doc = Document()
            doc.add(StringField("type", "node", Field.Store.YES))
            doc.add(StringField("id", str(node), Field.Store.YES))
            for key, value in data.items():
                doc.add(TextField(key, str(value), Field.Store.YES))
            writer.addDocument(doc)

        # Index edges
        for u, v, data in G.edges(data=True):
            doc = Document()
            doc.add(StringField("type", "edge", Field.Store.YES))
            doc.add(StringField("id", f"{u}-{v}", Field.Store.YES))
            doc.add(StringField("source", str(u), Field.Store.YES))
            doc.add(StringField("target", str(v), Field.Store.YES))
            for key, value in data.items():
                doc.add(TextField(key, str(value), Field.Store.YES))
            writer.addDocument(doc)

        writer.close()

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        reader = DirectoryReader.open(self.index_dir)
        searcher = IndexSearcher(reader)
        query = QueryParser("content", self.analyzer).parse(query)
        hits = searcher.search(query, limit)

        results = []
        for hit in hits.scoreDocs:
            doc = searcher.doc(hit.doc)
            item = {field.name(): field.stringValue() for field in doc.getFields()}
            results.append(item)

        reader.close()
        return results

    def get_node(self, node_id: str) -> Dict[str, Any]:
        reader = DirectoryReader.open(self.index_dir)
        searcher = IndexSearcher(reader)
        query = QueryParser("id", self.analyzer).parse(node_id)
        hits = searcher.search(query, 1)

        if len(hits.scoreDocs) == 0:
            reader.close()
            raise ValueError(f"Node with id {node_id} not found")

        doc = searcher.doc(hits.scoreDocs[0].doc)
        node = {field.name(): field.stringValue() for field in doc.getFields()}
        reader.close()
        return node

    def get_relation(self, relation_id: str) -> Dict[str, Any]:
        reader = DirectoryReader.open(self.index_dir)
        searcher = IndexSearcher(reader)
        query = QueryParser("id", self.analyzer).parse(relation_id)
        hits = searcher.search(query, 1)

        if len(hits.scoreDocs) == 0:
            reader.close()
            raise ValueError(f"Relation with id {relation_id} not found")

        doc = searcher.doc(hits.scoreDocs[0].doc)
        relation = {field.name(): field.stringValue() for field in doc.getFields()}
        reader.close()
        return relation
