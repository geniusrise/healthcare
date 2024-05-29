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
