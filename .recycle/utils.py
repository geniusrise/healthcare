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
from typing import List, Any

log = logging.getLogger(__name__)


def read_rrf_file(file_path: str, delimiter: str = "|") -> List[List[Any]]:
    """
    Reads an RRF (Rich Record Format) file and returns its contents as a list of rows.

    Args:
        file_path (str): Path to the RRF file.
        delimiter (str): Delimiter used in the file (default is '|').

    Returns:
        List of rows, where each row is a list of values.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=delimiter)
            return list(reader)
    except Exception as e:
        log.error(f"Error reading file {file_path}: {e}")
        raise ValueError(f"Error reading file {file_path}: {e}")
