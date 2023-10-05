import pytest
from geniusrise_healthcare.summary import (
    generate_summary,
)
from geniusrise_healthcare.model import load_huggingface_model

MODEL = "/run/media/ixaxaar/models_q/CodeLlama-34B-Python-GPTQ"

# Test data: list of conditions and questions-answers
test_conditions = [
    ["fatigue", "headache", "cough"],
    ["high blood pressure", "chest pain"],
]
test_qa = [
    {"How are you feeling?": "I'm feeling tired", "Do you have a cough?": "Yes"},
    {"Do you have chest pain?": "Yes", "How is your blood pressure?": "It's high"},
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
