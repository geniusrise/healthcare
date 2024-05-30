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

# utils.py
import logging
from pronto import Ontology

log = logging.getLogger(__name__)


def read_ontology_file(file_path: str) -> Ontology:
    """
    Reads an ontology file (OBO or OWL) and returns its Ontology object.

    Args:
        file_path (str): Path to the ontology file.

    Returns:
        Ontology object.
    """
    try:
        ontology = Ontology(file_path)
        return ontology
    except Exception as e:
        log.error(f"Error reading ontology file {file_path}: {e}")
        raise ValueError(f"Error reading ontology file {file_path}: {e}")
