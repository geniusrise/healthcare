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

# __init__.py
"""
UMLS NetworkX Loader Package

This package provides modules to load UMLS data into a NetworkX graph.
"""

from .attributes import process_attributes_file
from .base import load_umls_into_networkx
from .concepts import process_concepts_file
from .definitions import process_definitions_file
from .languages import process_languages_file
from .relationships import process_relationships_file
from .semantic_network import process_semantic_network_files
from .semantic_types import process_semantic_types_file
from .sources import process_sources_file
