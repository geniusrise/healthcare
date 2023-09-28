import csv
import logging
from typing import Dict

log = logging.getLogger(__name__)


def process_text_definition_file(text_definition_file: str, concept_id_to_text_definition: Dict[str, str]) -> None:
    """
    Processes the SNOMED CT text definition file and maps concept IDs to their text definitions.

    Parameters:
    - text_definition_file (str): Path to the text definition file.
    - concept_id_to_text_definition (Dict[str, str]): Dictionary to store the mapping from concept IDs to text definitions.

    Returns:
    None
    """
    file_length = 0
    with open(text_definition_file, "rbU") as f:
        num_lines = sum(1 for _ in f)

    log.info(f"Loading text definitions from {text_definition_file}")
    with open(text_definition_file, "r") as f:  # type: ignore
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)  # type: ignore
        next(reader)  # Skip header
        for row in reader:
            try:
                concept_id, active, term = row[4], row[2], row[7]
                if active == "1":
                    concept_id_to_text_definition[concept_id] = term
            except Exception as e:
                raise ValueError(f"Error processing text definition {row}: {e}")
