from typing import List, Tuple
from pyparsing import Word, nums, Literal, OneOrMore, Group, ZeroOrMore
import csv
import logging
import networkx as nx
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
    file_length = 0
    with open(owl_file, "rbU") as f:
        num_lines = sum(1 for _ in f)

    log.info(f"Loading OWL expressions from {owl_file}")
    with open(owl_file, "r") as f:  # type: ignore
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)  # type: ignore
        next(reader)  # Skip header
        for row in tqdm(reader, total=num_lines):
            try:
                owl_expression = row[6]
                referenced_component_id = row[4]
                edges = parse_owl_functional(owl_expression, referenced_component_id)
                for source, target, relationship_type in edges:
                    G.add_edge(source, target, relationship_type=relationship_type)
            except Exception as e:
                log.error(f"Error processing OWL expression {row}: {e}")
                raise ValueError(f"Error processing OWL expression {row}: {e}")
