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
import zipfile
from typing import Tuple

log = logging.getLogger(__name__)


def unzip_snomed_ct(zip_path: str, extract_path: str) -> None:
    log.info(f"Unzipping SNOMED-CT dataset from {zip_path} to {extract_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def extract_and_remove_semantic_tag(fsn: str) -> Tuple[str, str]:
    semantic_tag = fsn.split("(")[-1].rstrip(")").strip()
    fsn_without_tag = fsn.rsplit("(", 1)[0].strip()
    semantic_tag = "" if semantic_tag == fsn_without_tag else semantic_tag
    return semantic_tag, fsn_without_tag
