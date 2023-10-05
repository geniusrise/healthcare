import pytest
from geniusrise_healthcare.summary import (
    generate_summary,
)
from geniusrise_healthcare.model import load_huggingface_model

MODEL = "/run/media/ixaxaar/models_q/CodeLlama-34B-Python-GPTQ"

# Test data: list of conditions and questions-answers
test_conditions = [
    ["persistent cough", "fever", "fatigue"],
    ["chest pain", "shortness of breath", "dizziness"],
    ["abdominal pain", "vomiting", "diarrhea"],
    ["joint pain", "swelling", "redness"],
]

test_qa = [
    {
        "How long have you had these symptoms?": "About a week",
        "Are you experiencing any breathing difficulties?": "No",
        "Have you been in contact with anyone who has tested positive for COVID-19?": "Not that I'm aware of",
        "Do you have any pre-existing medical conditions?": "No",
        "Have you traveled recently?": "No",
        "On a scale of 1 to 10, how severe is your cough?": "6",
        "Are you experiencing loss of taste or smell?": "No",
    },
    {
        "How long have you been experiencing chest pain?": "A few hours",
        "Is the pain radiating to any other part of your body?": "It's going down my left arm",
        "Have you experienced this type of pain before?": "No",
        "Do you have a history of heart problems in your family?": "Yes, my father had a heart attack",
        "On a scale of 1 to 10, how severe is the pain?": "8",
        "Are you also feeling nauseous?": "Yes",
        "Have you taken any medication for the pain?": "No",
    },
    {
        "When did the abdominal pain start?": "Last night",
        "Is the pain constant or does it come and go?": "It comes and goes",
        "Have you eaten anything unusual recently?": "I had some seafood",
        "On a scale of 1 to 10, how severe is the pain?": "7",
        "Have you taken any medication for it?": "Just some antacids",
        "Are you experiencing any other symptoms like fever?": "No",
        "Have you vomited more than once?": "Yes, multiple times",
    },
    {
        "How long have you been experiencing joint pain?": "A couple of days",
        "Is the pain affecting multiple joints?": "Yes, mainly my knees and elbows",
        "Do you have a history of arthritis in your family?": "Yes, my mother has it",
        "On a scale of 1 to 10, how severe is the pain?": "5",
        "Are you able to move the affected joints?": "With difficulty",
        "Have you taken any medication for the pain?": "I took some ibuprofen",
        "Is the swelling getting worse?": "It's about the same",
    },
]

# Test data: for extract function
test_extract_data = [
    (
        "Here is a summary report of the above for the doctor to use as a string:\n\n```python\nSummary here\n```",
        "Summary here",
    ),
    ("No summary here", None),
]


@pytest.fixture(scope="module")
def load():
    model, tokenizer = load_huggingface_model(
        MODEL,
        use_cuda=True,
        precision="float16",
        quantize=False,
        quantize_bits=8,
        use_safetensors=True,
        trust_remote_code=True,
    )
    return tokenizer, model


@pytest.mark.parametrize("conditions,qa", zip(test_conditions, test_qa))
def test_generate_summary(conditions, qa, load):
    tokenizer, model = load
    result = generate_summary(
        tokenizer=tokenizer,
        model=model,
        conditions=conditions,
        qa=qa,
        decoding_strategy="generate",
        temperature=0.7,
        do_sample=True,
        max_new_tokens=200,
    )

    # Validate the result
    assert result["conditions"] == conditions
    assert result["qa"] == qa
    assert "summary" in result
    assert isinstance(result["summary"], str) or result["summary"] is None
    assert "speciality" in result
    assert isinstance(result["speciality"], str) or result["speciality"] is None
