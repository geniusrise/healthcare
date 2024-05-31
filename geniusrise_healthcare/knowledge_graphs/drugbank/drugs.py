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
from .utils import read_xml_file, get_elements_by_tag

log = logging.getLogger(__name__)


def process_drugs(drugbank_file: str, G: nx.DiGraph) -> None:
    """
    Processes the DrugBank drugs data and adds it to the graph.

    Args:
        drugbank_file (str): Path to the DrugBank XML file.
        G (nx.DiGraph): The NetworkX graph to which the drug data will be added.

    Returns:
        None
    """
    log.info(f"Loading drugs from {drugbank_file}")

    tree = read_xml_file(drugbank_file)
    drug_elements = get_elements_by_tag(tree, "drug")

    for drug in drug_elements:
        try:
            drugbank_id = drug.findtext("drugbank-id[@primary='true']")
            name = drug.findtext("name")
            description = drug.findtext("description")
            drug_type = drug.findtext("type")
            categories = [category.text for category in drug.findall("categories/category")]
            synonyms = [synonym.text for synonym in drug.findall("synonyms/synonym")]

            if drugbank_id:
                G.add_node(
                    drugbank_id,
                    name=name,
                    description=description,
                    type="drug",
                    drug_type=drug_type,
                    categories=categories,
                    synonyms=synonyms,
                )
        except Exception as e:
            log.error(f"Error processing drug {drug}: {e}")
            raise
