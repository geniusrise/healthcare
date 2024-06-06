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

# base.py
import logging
import networkx as nx

from disease_ontology.base import load_disease_ontology
from drugbank.base import load_drugbank
from gene_ontology.base import load_gene_ontology
from loinc.base import load_loinc
from mesh.base import load_mesh
from rxnorm.base import load_rxnorm
from snomed.base import load_snomed
from umls.base import load_umls

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load():
    G = nx.DiGraph()

    # for graph in graphs:

    # load_umls(G, "./data/umls/2024AA/META")
    # load_snomed(G, "./data/snomed/snomed/Snapshot/Terminology", version="INT_20240501")
    # load_rxnorm(G, "data/rxnorm/rrf")
    load_mesh(G, "./data/mesh")


if __name__ == "__main__":
    load()
