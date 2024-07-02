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
from typing import List, Dict, Set, Union
import asyncio
from collections import defaultdict
import logging
from base import load_graph as load

log = logging.getLogger(__name__)


class NetworkxAPI:
    def __init__(self, app):
        self.app = app
        self.graphs: Dict[str, nx.DiGraph] = {}
        self.pageranks: Dict[str, Dict[str, float]] = {}
        self.centralities: Dict[str, Dict[str, float]] = {}
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/load_graph/{graph_name}")
        async def load_graph(graph_name: str, compute_metrics: bool = False):
            try:
                self.graphs[graph_name] = await asyncio.to_thread(load, graph_name)
                if compute_metrics:
                    await self.precompute_metrics(graph_name)
                return {"message": f"Graph {graph_name} loaded successfully"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/search/{graph_name}")
        async def search_nodes(graph_name: str, query: str, limit: int = 10):
            graph = self.get_graph(graph_name)
            results = []
            for node, data in graph.nodes(data=True):
                if query.lower() in str(data).lower():
                    results.append({"id": node, "data": data})
                    if len(results) >= limit:
                        break
            return results

        @self.app.get("/traverse/{graph_name}/{node_id}")
        async def traverse_graph(graph_name: str, node_id: str, depth: int = 2):
            graph = self.get_graph(graph_name)
            if node_id not in graph:
                raise HTTPException(status_code=404, detail="Node not found")

            result = {node_id: {"data": dict(graph.nodes[node_id]), "neighbors": {}}}
            queue = [(node_id, result[node_id]["neighbors"], 0)]

            while queue:
                current_node, current_dict, current_depth = queue.pop(0)
                if current_depth >= depth:
                    continue
                for neighbor in graph.neighbors(current_node):
                    current_dict[neighbor] = {"data": dict(graph.nodes[neighbor]), "neighbors": {}}
                    queue.append((neighbor, current_dict[neighbor]["neighbors"], current_depth + 1))

            return result

        @self.app.get("/diffusion/{graph_name}/{node_id}")
        async def diffusion(graph_name: str, node_id: str, steps: int = 3):
            graph = self.get_graph(graph_name)
            if node_id not in graph:
                raise HTTPException(status_code=404, detail="Node not found")

            diffusion = defaultdict(float)
            diffusion[node_id] = 1.0

            for _ in range(steps):
                new_diffusion: defaultdict[str, float] = defaultdict(float)
                for node, value in diffusion.items():
                    neighbors = list(graph.neighbors(node))
                    if neighbors:
                        flow = value / len(neighbors)
                        for neighbor in neighbors:
                            new_diffusion[neighbor] += flow
                diffusion = new_diffusion

            return dict(diffusion)

        @self.app.get("/ranked_search/{graph_name}")
        async def ranked_search(graph_name: str, query: str, limit: int = 10):
            graph = self.get_graph(graph_name)
            pagerank = self.get_pagerank(graph_name)

            results = []
            for node, data in graph.nodes(data=True):
                if query.lower() in str(data).lower():
                    results.append({"id": node, "data": data, "rank": pagerank.get(node, 0)})

            results.sort(key=lambda x: x["rank"], reverse=True)
            return results[:limit]

        @self.app.get("/important_neighbors/{graph_name}/{node_id}")
        async def important_neighbors(graph_name: str, node_id: str, limit: int = 5):
            graph = self.get_graph(graph_name)
            centrality = self.get_centrality(graph_name)

            if node_id not in graph:
                raise HTTPException(status_code=404, detail="Node not found")

            neighbors = list(graph.neighbors(node_id))
            ranked_neighbors = sorted(neighbors, key=lambda n: centrality.get(n, 0), reverse=True)

            return [
                {"id": n, "data": dict(graph.nodes[n]), "centrality": centrality.get(n, 0)}
                for n in ranked_neighbors[:limit]
            ]

        @self.app.get("/local_important_nodes/{graph_name}/{node_id}")
        async def local_important_nodes(graph_name: str, node_id: str, n: int = 1):
            graph = self.get_graph(graph_name)
            important_nodes = self.find_local_important_nodes(graph, node_id, n)
            return [{"id": node, "data": dict(graph.nodes[node])} for node in important_nodes]

        @self.app.get("/recursive_search/{graph_name}/{node_id}")
        async def recursive_search_api(
            graph_name: str,
            node_id: str,
            semantic_types: List[str] = Query(None),
            stop_at_semantic_types: List[str] = Query(None),
            max_depth: int = 3,
        ):
            graph = self.get_graph(graph_name)
            top_one_percent_nodes = self.calculate_top_one_percent_nodes(graph)
            result_paths = self.recursive_search(
                graph, node_id, semantic_types, stop_at_semantic_types, set(), 0, max_depth, [], top_one_percent_nodes
            )
            return result_paths

        @self.app.get("/node_centrality/{graph_name}/{node_id}")
        async def node_centrality(graph_name: str, node_id: str):
            graph = self.get_graph(graph_name)
            centrality = self.get_centrality(graph_name)
            if node_id not in graph:
                raise HTTPException(status_code=404, detail="Node not found")
            return {"id": node_id, "centrality": centrality.get(node_id, 0)}

        @self.app.get("/shortest_path/{graph_name}")
        async def shortest_path(graph_name: str, source: str, target: str):
            graph = self.get_graph(graph_name)
            try:
                path = nx.shortest_path(graph, source, target)
                return [{"id": node, "data": dict(graph.nodes[node])} for node in path]
            except nx.NetworkXNoPath:
                raise HTTPException(status_code=404, detail="No path found between the specified nodes")

        @self.app.get("/common_neighbors/{graph_name}")
        async def common_neighbors(graph_name: str, node1: str, node2: str):
            graph = self.get_graph(graph_name)
            if node1 not in graph or node2 not in graph:
                raise HTTPException(status_code=404, detail="One or both nodes not found")
            common = list(nx.common_neighbors(graph, node1, node2))
            return [{"id": node, "data": dict(graph.nodes[node])} for node in common]

    def get_graph(self, graph_name: str) -> nx.DiGraph:
        if graph_name not in self.graphs:
            raise HTTPException(status_code=404, detail=f"Graph {graph_name} not loaded")
        return self.graphs[graph_name]

    def get_pagerank(self, graph_name: str) -> Dict[str, float]:
        if graph_name not in self.pageranks:
            raise HTTPException(status_code=404, detail=f"PageRank not computed for {graph_name}")
        return self.pageranks[graph_name]

    def get_centrality(self, graph_name: str) -> Dict[str, float]:
        if graph_name not in self.centralities:
            raise HTTPException(status_code=404, detail=f"Centrality not computed for {graph_name}")
        return self.centralities[graph_name]

    async def precompute_metrics(self, graph_name: str):
        graph = self.get_graph(graph_name)
        self.pageranks[graph_name] = await asyncio.to_thread(nx.pagerank, graph)
        self.centralities[graph_name] = await asyncio.to_thread(nx.betweenness_centrality, graph)

    def find_local_important_nodes(self, G: nx.DiGraph, node: str, n: int = 1) -> List[str]:
        subgraph = nx.ego_graph(G.to_undirected(), node, radius=n, undirected=True, center=True)
        return sorted(subgraph.nodes(), key=lambda x: subgraph.degree(x), reverse=True)

    def calculate_top_one_percent_nodes(self, G: nx.DiGraph) -> Set[str]:
        degrees = [(node, degree) for node, degree in G.degree()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        top_one_percent_count = max(1, int(len(degrees) * 0.01))
        return set(node for node, _ in degrees[:top_one_percent_count])

    def recursive_search(
        self,
        G: nx.DiGraph,
        node: str,
        semantic_types: Union[None, str, List[str]],
        stop_at_semantic_types: Union[None, str, List[str]],
        visited: Set[str],
        depth: int,
        max_depth: int,
        current_path: List[str],
        top_one_percent_nodes: Set[str],
    ) -> List[List[str]]:
        if node in visited or depth > max_depth:
            return []

        visited.add(node)
        current_path.append(node)
        paths = []

        node_tag = G.nodes[node].get("tag")
        if stop_at_semantic_types and node_tag in (
            stop_at_semantic_types if isinstance(stop_at_semantic_types, list) else [stop_at_semantic_types]
        ):
            paths.append(current_path.copy())
            current_path.pop()
            return paths

        if node in top_one_percent_nodes:
            paths.append(current_path.copy())
            current_path.pop()
            return paths

        if not semantic_types or node_tag in (semantic_types if isinstance(semantic_types, list) else [semantic_types]):
            paths.append(current_path.copy())

        candidates = list(G.predecessors(node)) + list(G.successors(node))
        for neighbor in candidates:
            paths += self.recursive_search(
                G,
                neighbor,
                semantic_types,
                stop_at_semantic_types,
                visited,
                depth + 1,
                max_depth,
                current_path,
                top_one_percent_nodes,
            )

        current_path.pop()
        return paths
