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
from typing import Dict
from tqdm import tqdm
from .utils import read_rrf_file

log = logging.getLogger(__name__)


def process_sources_file(sources_file: str, source_to_info: Dict[str, Dict]) -> None:
    """
    Processes the RxNorm source vocabulary file (RXNSAB.RRF) and stores the source information in a dictionary.

    Args:
        sources_file (str): Path to the RxNorm source vocabulary file (RXNSAB.RRF).
        source_to_info (Dict[str, Dict]): Dictionary to store the mapping of source abbreviations to source information.

    Returns:
        None
    """
    log.info(f"Loading sources from {sources_file}")

    rows = read_rrf_file(sources_file)
    for row in tqdm(rows):
        try:
            vcui, rcui, vsab, sver, rsab, son, sf, sver_date, scit, tcit = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
                row[9],
            )
            source_to_info[vsab] = {
                "vcui": vcui,
                "rcui": rcui,
                "sver": sver,
                "rsab": rsab,
                "son": son,
                "sf": sf,
                "sver_date": sver_date,
                "scit": scit,
                "tcit": tcit,
            }
        except Exception as e:
            log.error(f"Error processing source {row}: {e}")
            raise ValueError(f"Error processing source {row}: {e}")
