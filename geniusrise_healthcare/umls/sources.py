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


def process_sources_file(sources_file: str, source_to_info: Dict[str, Dict]) -> None:
    """
    Processes the UMLS source vocabulary file (MRSAB.RRF) and stores the source information in a dictionary.

    Args:
        sources_file (str): Path to the UMLS source vocabulary file (MRSAB.RRF).
        source_to_info (Dict[str, Dict]): Dictionary to store the mapping of source abbreviations to source information.

    Returns:
        None
    """
    log.info(f"Loading sources from {sources_file}")

    with open(sources_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in tqdm(reader):
            try:
                sab, description = row[1], row[2]
                source_to_info[sab] = {"description": description}
            except Exception as e:
                log.error(f"Error processing source {row}: {e}")
                raise ValueError(f"Error processing source {row}: {e}")
