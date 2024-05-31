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
import xml.etree.ElementTree as ET
from typing import List

log = logging.getLogger(__name__)


def read_xml_file(file_path: str) -> ET.ElementTree:
    """
    Reads an XML file and returns its ElementTree.

    Args:
        file_path (str): Path to the XML file.

    Returns:
        ElementTree of the XML file.
    """
    try:
        tree = ET.parse(file_path)
        return tree
    except Exception as e:
        log.error(f"Error reading XML file {file_path}: {e}")
        raise ValueError(f"Error reading XML file {file_path}: {e}")


def get_elements_by_tag(tree: ET.ElementTree, tag: str) -> List[ET.Element]:
    """
    Gets all elements by tag from an ElementTree.

    Args:
        tree (ET.ElementTree): The ElementTree.
        tag (str): The tag of the elements to find.

    Returns:
        List of elements with the specified tag.
    """
    return tree.findall(f".//{tag}")
