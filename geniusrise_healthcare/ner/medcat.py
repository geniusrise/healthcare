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

import logging
from typing import Dict, List, Union

import pandas as pd
from medcat.cat import CAT

log = logging.getLogger(__name__)


def annotate_snomed(
    cat: CAT, data: pd.DataFrame, type_ids_filter: List[str], page_size: int = 5000
) -> Dict[int, Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]]:
    """
    Annotate text with SNOMED concept IDs using MedCAT.

    Parameters:
    - cat (CAT): The MedCAT Clinical Annotation Tool instance.
    - data (pd.DataFrame): The DataFrame containing text to be annotated.
    - type_ids_filter (List[str]): List of type IDs to filter annotations.
    - page_size (Optional[int]): The size of each text chunk for pagination. Default is 5000 characters.

    Returns:
    - Dict[int, Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]]: A dictionary containing the document ID and its annotations.

    Usage:
    ```python
    cat = CAT.load_model_pack("path/to/model_pack")
    data = pd.read_csv("path/to/data.csv")
    type_ids_filter = ['T047', 'T048']
    results = annotate_snomed(cat, data, type_ids_filter)
    ```

    """
    # Initialize results dictionary
    results: Dict[int, Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]] = {}

    try:
        # Filter CDB to only include the provided type IDs
        cui_filters = set()
        for type_id in type_ids_filter:
            cui_filters.update(cat.cdb.addl_info["type_id2cuis"][type_id])
        cat.cdb.config.linking["filters"]["cuis"] = cui_filters

        # Iterate over the DataFrame
        for doc_id, row in data.iterrows():
            text = str(row["text"])
            annotations: List[Dict[str, Union[str, int]]] = []

            # Implement pagination for large texts
            for i in range(0, len(text), page_size):
                chunk = text[i : i + page_size]
                log.debug(f"Annotating chunk {i // page_size + 1} of document {doc_id}")

                # Extract annotations
                entities = cat.get_entities(chunk)
                for ann in entities["entities"].values():
                    annotations.append(
                        {
                            "start": ann["start"],
                            "end": ann["end"],
                            "text": ann["name"],
                            "cui": ann["cui"],
                            "type_id": ann["type_id"],
                            "types": ann["types"],
                            "snomed": ann["snomed"],
                        }
                    )
                    log.info(f"Detected entity {annotations}")

            # Store the result
            results[doc_id] = {"text": text, "annotations": annotations}

        log.info("Annotation process completed.")
        return results

    except Exception as e:
        log.exception(f"An error occurred: {e}")
        raise
