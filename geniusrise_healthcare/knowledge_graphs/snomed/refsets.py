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

import csv
import logging
from typing import List, Tuple

import networkx as nx
from pyparsing import Group, Literal, OneOrMore, Word, ZeroOrMore, nums
from tqdm import tqdm

log = logging.getLogger(__name__)

# Define OWL Functional Language Grammar using pyparsing
integer = Word(nums)
colon = Literal(":")
owl_class = colon + integer
sub_class_of = Literal("SubClassOf")
equivalent_classes = Literal("EquivalentClasses")
object_intersection_of = Literal("ObjectIntersectionOf")
object_some_values_from = Literal("ObjectSomeValuesFrom")
sub_object_property_of = Literal("SubObjectPropertyOf")
transitive_object_property = Literal("TransitiveObjectProperty")
sub_data_property_of = Literal("SubDataPropertyOf")
reflexive_object_property = Literal("ReflexiveObjectProperty")
prefix = Literal("Prefix")
ontology = Literal("Ontology")

# Update the grammar to include the new constructs
owl_expression_grammar = OneOrMore(
    Group(
        sub_class_of
        | equivalent_classes
        | object_intersection_of
        | object_some_values_from
        | sub_object_property_of
        | transitive_object_property
        | sub_data_property_of
        | reflexive_object_property
        | prefix
        | ontology
    )
    + ZeroOrMore(Group(owl_class))
)


def parse_owl_functional(owl_expression: str, referenced_component_id: str) -> List[Tuple[int, int, str]]:
    """
    Parses an OWL functional expression and returns the edges for the graph.

    Args:
        owl_expression (str): The OWL functional expression.
        referenced_component_id (str): The component ID referenced in the expression.

    Returns:
        List of tuples representing the edges to be added to the graph.
    """
    try:
        parsed = owl_expression_grammar.parseString(owl_expression)
        edges = []
        relationship_type = parsed[0][0]

        for item in parsed[1:]:
            if item[0].startswith(":"):
                target = int(item[0][1:])
                source = int(referenced_component_id)
                edges.append((source, target, relationship_type))

        return edges
    except Exception as e:
        log.error(f"Error parsing OWL expression {owl_expression}: {e}")
        raise ValueError(f"Error parsing OWL expression {owl_expression}: {e}")


def process_refsets_file(refsets_file: str, G: nx.DiGraph) -> None:
    """
    Processes the SNOMED CT OWL refsets file and adds the relationships to the graph.

    Args:
        refsets_file (str): Path to the OWL refsets file.
        G (nx.DiGraph): The NetworkX graph to which the relationships will be added.

    Returns:
        None
    """
    log.info(f"Loading OWL refsets from {refsets_file}")

    with open(refsets_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip header
        for row in tqdm(reader):
            try:
                (id, effective_time, active, module_id, refset_id, referenced_component_id, owl_expression) = row[:7]
                if active == "1" and referenced_component_id in G:
                    if "owl_axioms" not in G.nodes[referenced_component_id]:
                        G.nodes[referenced_component_id]["owl_axioms"] = []
                    G.nodes[referenced_component_id]["owl_axioms"].append(
                        {"id": id, "refset_id": refset_id, "owl_expression": owl_expression}
                    )
            except Exception as e:
                log.error(f"Error processing OWL refset {row}: {e}")
                raise ValueError(f"Error processing OWL refset {row}: {e}")
