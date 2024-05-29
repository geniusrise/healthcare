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

from .llama import annotate_snomed as annotate_snomed_llama2
from .medcat import annotate_snomed as annotate_snomed_medcat


def annotate_snomed(type: str, *args, **kwargs):
    if type == "medcat":
        return annotate_snomed_medcat(*args, **kwargs)
    elif type == "llm":
        return annotate_snomed_llama2(*args, **kwargs)
    else:
        raise ValueError(f"This type of annotator is not supported: {type}")
