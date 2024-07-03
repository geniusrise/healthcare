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

from disease_ontology.base import load_disease_ontology
from gene_ontology.base import load_gene_ontology
from mesh.base import load_mesh
from rxnorm.base import load_rxnorm
from snomed.base import load_snomed
from umls.base import load_umls

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_graph(name: str) -> nx.DiGraph:
    """
    Load a knowledge graph with networkx.

    Returns:
        Dict[str, nx.DiGraph]: A dictionary of the loaded graphs.
    """
    if name == "umls":
        return load_umls(nx.DiGraph(), "../../data/umls/2024AA/META")

    if name == "snomed":
        return load_snomed(nx.DiGraph(), "../../data/snomed/snomed/Snapshot/Terminology", version="INT_20240501")
    elif name == "rxnorm":
        return load_rxnorm(nx.DiGraph(), "../../data/rxnorm/rrf")
    elif name == "mesh":
        return load_mesh(nx.DiGraph(), "../../data/mesh")
    elif name == "gene_ontology":
        return load_gene_ontology(nx.DiGraph(), "../../data/gene_ontology/go.owl")
    elif name == "disease_ontology":
        return load_disease_ontology(
            nx.DiGraph(), "../../data/disease_ontology/HumanDiseaseOntology/src/ontology/releases/doid-merged.owl"
        )
    else:
        raise ValueError("Requested unrecognized graph")
