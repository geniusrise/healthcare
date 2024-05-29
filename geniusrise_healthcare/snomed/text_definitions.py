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
from typing import Dict
from tqdm import tqdm

log = logging.getLogger(__name__)


def process_text_definition_file(text_definition_file: str, concept_id_to_text_definition: Dict[str, str]) -> None:
    """
    Processes the SNOMED CT text definition file and maps concept IDs to their text definitions.

    Args:
        text_definition_file (str): Path to the text definition file.
        concept_id_to_text_definition (Dict[str, str]): Dictionary to store the mapping from concept IDs to text definitions.

    Returns:
        None
    """
    with open(text_definition_file, "rbU") as f:
        num_lines = sum(1 for _ in f)

    log.info(f"Loading text definitions from {text_definition_file}")
    with open(text_definition_file, "r") as f:  # type: ignore
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)  # type: ignore
        next(reader)
        for row in tqdm(reader, total=num_lines):
            try:
                concept_id, active, term, definition_type, case_significance = (
                    row[4],
                    row[2],
                    row[7],
                    row[6],
                    row[8],
                )
                if active == "1":
                    concept_id_to_text_definition[concept_id] = {  # type: ignore
                        "term": term,
                        "definition_type": definition_type,
                        "case_significance": case_significance,
                    }
            except Exception as e:
                log.error(f"Error processing text definition {row}: {e}")
                raise ValueError(f"Error processing text definition {row}: {e}")
