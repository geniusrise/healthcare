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

# | Top level hierarchy                  | Sub hierarchy tag         | 2 tag  |
# | ------------------------------------ | ------------------------- | ------ |
# | Clinical Finding                     | (disorder)                |        |
# |                                      | (finding)                 |        |
# | Procedure                            | (procedure)               |        |
# |                                      | (regime/therapy)          |        |
# | Event                                | (event)                   |        |
# | Observable Entity                    | (observable entity)       |        |
# | Situation with explicit context      | (situation)               |        |
# | Pharmaceutical / biologic product    | (product)                 |        |
# | Social Context                       | (social concept)          |        |
# |                                      | (person)                  |        |
# |                                      | (ethnic group)            |        |
# |                                      | (racial group)            |        |
# |                                      | (religion/philosophy)     |        |
# |                                      | (occupation)              |        |
# |                                      | (life style)              |        |
# |                                      | (family)                  |        |
# | Body Structure                       | (body structure)          | (cell) |
# |                                      | (morphologic abnormality) |        |
# | Organism                             | (organism)                |        |
# | Physical Object                      | (physical object)         |        |
# | Substance                            | (substance)               |        |
# | Specimen                             | (specimen)                |        |
# | Physical Force                       | (physical force)          |        |
# | Environment or geographical location | (environment)             |        |
# |                                      | (geographic location)     |        |
# | Staging and Scales                   | (assessment scale)        |        |
# |                                      | (tumor staging)           |        |
# |                                      | (staging scale)           |        |
# | Qualifier value                      | (qualifier value)         |        |
# | Linkage concept                      | (attribute)               |        |
# |                                      | (link assertion)          |        |
# | Special concept                      | (inactive concept)        |        |
# |                                      | (namespace concept)       |        |
# |                                      | (navigational concept)    |        |
# | Record artifact                      | (record artifact)         |        |

SEMANTIC_TAGS = [
    "body structure",
    "cell",
    "cell structure",
    "morphologic abnormality",
    "finding",
    "disorder",
    "environment",
    "geographic location",
    "event",
    "observable entity",
    "organism",
    "clinical drug",
    "medicinal product",
    "medicinal product form",
    "physical object",
    "product",
    "physical force",
    "procedure",
    "regime/therapy",
    "qualifier value",
    "administration method",
    "basic dose form",
    "disposition",
    "dose form",
    "intended site",
    "number",
    "product name",
    "release characteristic",
    "role",
    "state of matter",
    "transformation",
    "supplier",
    "unit of presentation",
    "record artifact",
    "situation",
    "attribute",
    "core metadata concept",
    "foundation metadata concept",
    "link assertion",
    "linkage concept",
    "namespace concept",
    "OWL metadata concept",
    "social concept",
    "ethnic group",
    "life style",
    "occupation",
    "person",
    "racial group",
    "religion/philosophy",
    "inactive concept",
    "navigational concept",
    "specimen",
    "staging scale",
    "assessment scale",
    "tumor staging",
    "substance",
]
