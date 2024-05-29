# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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


def process_refsets_file(owl_file: str, G: nx.DiGraph) -> None:
    """
    Processes the SNOMED CT OWL refsets file and adds the relationships to the graph.

    Args:
        owl_file (str): Path to the OWL refsets file.
        G (nx.DiGraph): The NetworkX graph to which the relationships will be added.

    Returns:
        None
    """
    with open(owl_file, "rbU") as f:
        num_lines = sum(1 for _ in f)

    log.info(f"Loading OWL expressions from {owl_file}")
    with open(owl_file, "r") as f:  # type: ignore
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)  # type: ignore
        next(reader)
        for row in tqdm(reader, total=num_lines):
            try:
                owl_expression, referenced_component_id, refset_id, reference_component = (
                    row[6],
                    row[4],
                    row[5],
                    row[8],
                )
                edges = parse_owl_functional(owl_expression, referenced_component_id)
                for source, target, relationship_type in edges:
                    G.add_edge(
                        source,
                        target,
                        relationship_type=relationship_type,
                        refset_id=refset_id,
                        reference_component=reference_component,
                    )
            except Exception as e:
                log.error(f"Error processing OWL expression {row}: {e}")
                raise ValueError(f"Error processing OWL expression {row}: {e}")
