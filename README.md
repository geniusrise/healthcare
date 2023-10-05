![banner](./assets/banner.png)

# Geniusrise Healthcare APIs

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
  "symptoms_diseases": ["cold", "fever", "headache"],
  "snomed_concept_ids": [
    [84162001, 313094006, 82272006],
    [248425001, 386661006],
    [25064002, 139490008, 162209005, 224973000, 257553007, 191667009]
  ],
  "snomed_concepts": [
    ["cold sensation quality", "cold - thermal agent", "common cold"],
    ["pyrexia", "febrile"],
    [
      "pain in head",
      "headache",
      "headache (& [c/o])",
      "feeling frustrated",
      "irritation",
      "paranoid psychosis"
    ]
  ]
}
```

### Generate follow up questions

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
                ],
                [
                  25064002,
                  139490008,
                  162209005,
                  224973000,
                  257553007,
                  191667009
                ]
              ]
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
      "Do you have a sore throat?",
      "Do you have a headache?",
      "Do you have a runny nose?"
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
      "Do you take any medications?",
      "What is your age?"
    ]
  },
  {
    "snomed_concept_ids": [
      25064002,
      139490008,
      162209005,
      224973000,
      257553007,
      191667009
    ],
    "snomed_concepts": [
      "pain in head",
      "headache",
      "headache (& [c/o])",
      "feeling frustrated",
      "irritation",
      "paranoid psychosis"
    ],
    "questions": [
      "Could you describe your pain in your head in more detail, perhaps using an example?",
      "Is this a dull ache across your forehead, or perhaps a throbbing pain?",
      "Do you think your headache is caused by lack of sleep?",
      "Are you more frustrated than usual?",
      "Do you feel like you're losing control of yourself?"
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
  "summary": "# In-Patient Report\n\n## Summary\n\n### Patient's Observations\n\nThe patient presented with the following complaints:\n\n- cold sensation quality\n- cold - thermal agent\n- common cold\n- pyrexia\n- febrile\n\n### Questions and Answers\n\nSubsequently, on further questioning, the patient had these to add on:\n\n- **Question**: How long did the cough last? \n  - **Answer**: its still there\n\n- **Question**: How long have you had the cold or cough? \n  - **Answer**: for the last 5 days\n\n- **Question**: Has the temperature continued to rise? \n  - **Answer**: yes\n\n- **Question**: Has the temperature stayed the same? \n  - **Answer**: no it keeps changing mostly it has risen\n\n## Recommended tests\n\n### Tests for diagnosis\n\n- Chest X-ray\n- Blood tests\n\n### Tests for exclusion\n\n- PET-CT\n\n## Diseases to check for and narrow down the cause\n\n- Bronchitis\n- Rhinitis\n- Upper Respiratory Infection\n- Chronic Obstructive Pulmonary Disease\n- Sinusitis",
  "speciality": "### Department\n\nThe patient should visit the **Outpatient** department."
}
```

### Generate snomed diagram

```bash
http POST http://localhost:2180/generate_snomed_graph snomed_concepts:='[[84162001, 313094006, 82272006], [248425001, 386661006]]' --download
```

```
Content-Type: multipart/mixed; boundary="===============0359621490805120141=="
MIME-Version: 1.0

--===============0359621490805120141==
Content-Type: text/plain; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit

Graph:
like ice - sensation quality --[is a]--> cold sensation quality
ice cold - sensation --[is a]--> cold sensation quality
puo - pyrexia of unknown origin --[is a]--> febrile
fever greater than 38 celsius --[is a]--> febrile
cebpe-associated autoinflammation, immunodeficiency, neutrophil dysfunction syndrome --[associated with]--> febrile
fever with rigors --[is a]--> febrile
fever caused by virus --[is a]--> febrile
febrile transfusion reaction --[is a]--> febrile
education about managing fever --[has focus]--> febrile
low grade fever --[is a]--> febrile
nlrc4-related autoinflammatory syndrome with macrophage activation syndrome --[associated with]--> febrile
fever of newborn --[is a]--> febrile
thrombocytopenia, anasarca, fever, renal insufficiency, organomegaly syndrome --[is a]--> febrile
postpartum fever --[is a]--> febrile
chronic fever --[is a]--> febrile
finding of pattern of fever --[is a]--> febrile
paraneoplastic fever --[is a]--> febrile
cough with fever --[is a]--> febrile
fever caused by severe acute respiratory syndrome coronavirus 2 --[is a]--> febrile
febrile proteinuria --[associated with]--> febrile
aseptic fever --[is a]--> febrile
postprocedural fever --[is a]--> febrile
maternal pyrexia --[associated finding]--> febrile
maternal pyrexia during labor --[is a]--> febrile
fever due to infection --[is a]--> febrile
treatment of fever --[has focus]--> febrile
sweating fever --[is a]--> febrile
fever-associated acute infantile liver failure syndrome --[following]--> febrile
hyperpyrexia --[is a]--> febrile
bancroftian filarial fever --[is a]--> febrile
apyrexial --[associated finding]--> febrile
malayan filarial fever --[is a]--> febrile
sweet's disease caused by drug --[associated with]--> febrile

--===============0359621490805120141==
Content-Type: application/octet-stream; Name="image.png"
MIME-Version: 1.0
Content-Transfer-Encoding: base64

iVBORw0KGgoAAAANSUhEUgAAC/QAAAv0CAYAAACNCdhAAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90
bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAC4j

--===============0359621490805120141==--
```
