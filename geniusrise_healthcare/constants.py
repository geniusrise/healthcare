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
