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
genius InPatientAPI rise\
    batch \
        --input_s3_bucket geniusrise-test-healthcare \
        --input_s3_folder snomed-graph \
    batch \
        --output_s3_bucket geniusrise-test-healthcare \
        --output_s3_folder model \
    none \
    listen \
        --args \
            endpoint="*" \
            port=2180 \
            llm_model="/run/media/ixaxaar/models_q/CodeLlama-34B-Python-GPTQ" \
            ner_model="emilyalsentzer/Bio_ClinicalBERT" \
            networkx_graph="./saved/snomed.graph" \
            faiss_index="./saved/faiss.index.Bio_ClinicalBERT" \
            concept_id_to_concept="./saved/concept_id_to_concept.pickle" \
            description_id_to_concept="./saved/description_id_to_concept.pickle"
```

## APIs

### NER symptoms and diseases

```bash
curl -s -X POST \
     -H "Content-Type: application/json" \
     -d '{"user_input": "i feel a bit light headed and have some difficulty breathing and some chest pain"}' \
     http://localhost:2180/find_symptoms_diseases | jq
```

```json
{
  "query": "i feel a bit light headed and have some difficulty breathing and some chest pain",
  "symptoms_diseases": ["light headed", "difficulty breathing", "chest pain"],
  "snomed_concept_ids": [
    [139200001, 161945003, 230145002],
    [139228007, 29857009]
  ],
  "snomed_concepts": [
    ["difficulty breathing", "difficulty breathing", "difficulty breathing"],
    ["chest pain", "chest pain"]
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
              139200001,
              161945003,
              230145002
            ],
            [
              139228007,
              29857009
            ]
          ],
          "symptoms_diseases": [
            "light headed",
            "difficulty breathing",
            "chest pain"
          ]
         }' \
     http://localhost:2180/generate_follow_up_questions_from_concepts | jq
```

```bash
[
  {
    "snomed_concept_ids": [
      139200001,
      161945003,
      230145002
    ],
    "snomed_concepts": [
      "difficulty breathing",
      "difficulty breathing",
      "difficulty breathing"
    ],
    "questions": [
      "Do you have a cough?",
      "Do you have a fever?",
      "Have you recently traveled abroad?"
    ]
  },
  {
    "snomed_concept_ids": [
      139228007,
      29857009
    ],
    "snomed_concepts": [
      "chest pain",
      "chest pain"
    ],
    "questions": [
      "Do you feel like you're having a heart attack?",
      "Does this feel like a heart attack?"
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
              139200001,
              161945003,
              230145002
            ],
            [
              139228007,
              29857009
            ]
        ],
        "symptoms_diseases": [
          "light headed",
          "difficulty breathing",
          "chest pain"
        ],
        "qa": {
          "Do you have a cough?": "no",
          "Do you have a fever?": "no",
          "Have you recently traveled abroad?": "no",
          "Do you feel like youre having a heart attack?": "i dont know",
          "Does this feel like a heart attack?": "maybe i never had one before so dont know"
        }
    }' \
    http://localhost:2180/generate_summary_from_qa | jq
```

```json
{
  "conditions": [
    "difficulty breathing",
    "difficulty breathing",
    "difficulty breathing",
    "chest pain",
    "chest pain"
  ],
  "qa": {
    "Do you have a cough?": "no",
    "Do you have a fever?": "no",
    "Have you recently traveled abroad?": "no",
    "Do you feel like youre having a heart attack?": "i dont know",
    "Does this feel like a heart attack?": "maybe i never had one before so dont know"
  },
  "summary": "# In-Patient Report

  ## Summary

  ### Patient's Observations

  The patient presented with the following complaints:

  ['light headed', 'difficulty breathing', 'chest pain']

  of which these are the conditions identified from SNOMED:

  - difficulty breathing
  - difficulty breathing
  - difficulty breathing
  - chest pain
  - chest pain

  ### Questions and Answers

  Subsequently, on further questioning, the patient had these to add on:

  - **Question**: Do you have a cough?
    - **Answer**: no

  - **Question**: Do you have a fever?
    - **Answer**: no

  - **Question**: Have you recently traveled abroad?
    - **Answer**: no

  - **Question**: Do you feel like youre having a heart attack?
    - **Answer**: i dont know

  - **Question**: Does this feel like a heart attack?
    - **Answer**: maybe i never had one before so dont know

  ## Recommended tests


  ### Tests for diagnosis

  - Serum electrolytes
  - Blood count
  - chest x-ray

  ### Tests for exclusion

  - Serum electrolytes
  - Blood count
  - chest x-ray


  ## Diseases to check for and narrow down the cause

  - Pulmonary embolism
  - Acute coronary syndrome
  - Acute exacerbation of COPD
  - Acute respiratory distress syndrome
  - Acute chest syndrome
  - Acute exacerbation of asthma
  - Myocardial infarction
  - Acute decompensated heart failure
  - Acute exacerbation of heart failure
  - Acute myocarditis
  - Acute myocarditis",
  "speciality": "### Department

  The patient should visit the **Outpatient** department."
}
```

### Generate snomed diagram

```bash
http POST http://localhost:2180/generate_snomed_graph snomed_concepts:='[
            [
              139200001,
              161945003,
              230145002
            ],
            [
              139228007,
              29857009
            ]
        ]' --download
```

```
Content-Type: multipart/mixed; boundary="===============0359621490805120141=="
MIME-Version: 1.0

--===============0359621490805120141==
Content-Type: text/plain; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit

Graph:
dyspnea --[is a]--> difficulty breathing
winded --[is a]--> difficulty breathing
labored breathing --[is a]--> difficulty breathing
respiratory distress --[is a]--> difficulty breathing
cannot blow --[is a]--> difficulty breathing
radiating chest pain --[is a]--> chest pain
microvascular angina --[is a]--> chest pain
pain of breast --[is a]--> chest pain
parasternal pain --[is a]--> chest pain
right sided chest pain --[is a]--> chest pain
assessment of chest pain --[has focus]--> chest pain
left sided chest pain --[is a]--> chest pain
thoracic back pain --[is a]--> chest pain
burning chest pain --[is a]--> chest pain
ischemic chest pain --[is a]--> chest pain
chest pain not present --[associated finding]--> chest pain
chest pain due to pericarditis --[is a]--> chest pain
crushing chest pain --[is a]--> chest pain
chest pain at rest --[is a]--> chest pain
localized chest pain --[is a]--> chest pain
retrosternal pain --[is a]--> chest pain
history of chest pain --[associated finding]--> chest pain
dull chest pain --[is a]--> chest pain
chest pain on breathing --[is a]--> chest pain
chronic chest pain --[is a]--> chest pain
pleuritic pain --[is a]--> chest pain
atypical chest pain --[is a]--> chest pain
musculoskeletal chest pain --[is a]--> chest pain
intercostal neuralgia --[is a]--> chest pain
upper chest pain --[is a]--> chest pain
angina --[is a]--> chest pain
noncardiac chest pain --[is a]--> chest pain
chest wall pain --[is a]--> chest pain
cardiac chest pain --[is a]--> chest pain
pleuropericardial chest pain --[is a]--> chest pain
precordial pain --[is a]--> chest pain
chest pain on exertion --[is a]--> chest pain
squeezing chest pain --[is a]--> chest pain
acute chest pain --[is a]--> chest pain
esophageal chest pain --[is a]--> chest pain

--===============0359621490805120141==
Content-Type: application/octet-stream; Name="image.png"
MIME-Version: 1.0
Content-Transfer-Encoding: base64

iVBORw0KGgoAAAANSUhEUgAAC/QAAAv0CAYAAACNCdhAAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90
bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAC4j

--===============0359621490805120141==--
```
