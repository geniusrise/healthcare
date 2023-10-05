![banner](./assets/banner.png)

# Geniusrise Healthcare Modules

1. NER using vector knowledge graph search
2. Semantic similarity to find a consistent set of related conditions, diseases,
   etc
3. Graph-based measures to find a consistent set of related conditions,
   diseases, etc
4. Various combinations of the above

# Usage

```bash
LOGLEVEL=DEBUG genius InPatientAPI rise\
    batch \
        --input_s3_bucket geniusrise-test \
        --input_s3_folder train \
    batch \
        --output_s3_bucket geniusrise-test \
        --output_s3_folder model \
    none \
    listen \
        --args \
            endpoint="*" \
            port=2180 \
            llm_model="/run/media/ixaxaar/models_q/CodeLlama-34B-Python-GPTQ" \
            ner_model="bert-base-uncased" \
            networkx_graph="./saved/snomed.graph" \
            faiss_index="./saved/faiss.index" \
            concept_id_to_concept="./saved/concept_id_to_concept.pickle" \
            description_id_to_concept="./saved/description_id_to_concept.pickle"
```

## APIs

### NER symptoms and diseases

```bash
curl -s -X POST \
     -H "Content-Type: application/json" \
     -d '{"user_input": "i have been feeling a bit cold and i think i have a fever and a light headache"}' \
     http://localhost:2180/find_symptoms_diseases | jq
```

```json
{
  "query": "i have been feeling a bit cold and i think i have a fever and a light headache",
  "symptoms_diseases": ["cold", "fever", "light headache"],
  "snomed_concept_ids": [
    [84162001, 313094006, 82272006],
    [248425001, 386661006]
  ],
  "snomed_concepts": [
    ["cold sensation quality", "cold - thermal agent", "common cold"],
    ["pyrexia", "febrile"]
  ]
}
```

### Generate follow up questions

```bash
curl -s -X POST \
     -H "Content-Type: application/json" \
     -d '{
           "snomed_concept_ids": [[84162001, 313094006, 82272006], [248425001, 386661006]]
         }' \
     http://localhost:2180/generate_follow_up_questions_from_concepts | jq
```

```bash
[
  {
    "snomed_concept_ids": [
      84162001,
      313094006,
      82272006
    ],
    "snomed_concepts": [
      "cold sensation quality",
      "cold - thermal agent",
      "common cold"
    ],
    "questions": [
      "Is it a runny nose?",
      "How long did the cough last?",
      "How long have you had the cold or cough?"
    ]
  },
  {
    "snomed_concept_ids": [
      248425001,
      386661006
    ],
    "snomed_concepts": [
      "pyrexia",
      "febrile"
    ],
    "questions": [
      "Has the temperature continued to rise?",
      "Has the temperature stayed the same?"
    ]
  }
]
```

### Generate final summary report

```bash
curl -s -X POST \
     -H "Content-Type: application/json" \
     -d '{
        "snomed_concept_ids": [
            [
                84162001,
                313094006,
                82272006
            ],
            [
                248425001,
                386661006
            ]
        ],
        "qa": {
            "How long did the cough last?": "its still there",
            "How long have you had the cold or cough?": "for the last 5 days",
            "Has the temperature continued to rise?": "yes",
            "Has the temperature stayed the same?": "no it keeps changing mostly it has risen"
        }
    }' \
    http://localhost:2180/generate_summary_from_qa | jq
```

```bash
{
  "conditions": [
    "cold sensation quality",
    "cold - thermal agent",
    "common cold",
    "pyrexia",
    "febrile"
  ],
  "qa": {
    "How long did the cough last?": "its still there",
    "How long have you had the cold or cough?": "for the last 5 days",
    "Has the temperature continued to rise?": "yes",
    "Has the temperature stayed the same?": "no it keeps changing mostly it has risen"
  },
  "summary": "# In-Patient Report\n\n## Summary\n\n### Patient's Observations\n\nThe patient presented with the following complaints:\n\n- cold sensation quality\n- cold - thermal agent\n- common cold\n- pyrexia\n- febrile\n\n### Questions and Answers\n\nSubsequently, on further questioning, the patient had these to add on:\n\n- **Question**: How long did the cough last? \n  - **Answer**: its still there\n\n- **Question**: How long have you had the cold or cough? \n  - **Answer**: for the last 5 days\n\n- **Question**: Has the temperature continued to rise? \n  - **Answer**: yes\n\n- **Question**: Has the temperature stayed the same? \n  - **Answer**: no it keeps changing mostly it has risen\n\n## Recommended tests\n\n### Tests for diagnosis\n\n- Complete blood count\n\n### Tests for exclusion\n\n- X-Ray scan\n\n## Diseases to check for and narrow down the cause\n\n- Adenovirus\n- Influenza A\n- Streptococcal pharyngitis",
  "speciality": "## Speciality\n\n## Department"
}
```

### Generate snomed diagram

```bash
curl -s -X POST \
     -H "Content-Type: application/json" \
     -d '{
        "snomed_concepts": [
            [
                84162001,
                313094006,
                82272006
            ],
            [
                248425001,
                386661006
            ]
        ]
    }' \
    http://localhost:2180/generate_snomed_graph | jq
```
