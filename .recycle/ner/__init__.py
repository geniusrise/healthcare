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

from .llama import annotate_snomed as annotate_snomed_llama2
from .medcat import annotate_snomed as annotate_snomed_medcat


def annotate_snomed(type: str, *args, **kwargs):
    if type == "medcat":
        return annotate_snomed_medcat(*args, **kwargs)
    elif type == "llm":
        return annotate_snomed_llama2(*args, **kwargs)
    else:
        raise ValueError(f"This type of annotator is not supported: {type}")
