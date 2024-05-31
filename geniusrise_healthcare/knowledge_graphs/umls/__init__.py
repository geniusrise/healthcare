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

from .attributes import process_attributes_file
from .base import load_umls_into_networkx
from .concepts import process_concepts_file
from .definitions import process_definitions_file
from .languages import process_languages_file
from .relationships import process_relationships_file
from .semantic_network import process_semantic_network_files
from .semantic_types import process_semantic_types_file
from .sources import process_sources_file
