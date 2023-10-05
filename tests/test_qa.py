import pytest

from geniusrise_healthcare.model import load_huggingface_model
from geniusrise_healthcare.qa import generate_follow_up_questions

# MODEL = "/run/media/ixaxaar/hynix_2tb/models/Llama-2-7b-hf"
# MODEL = "/run/media/ixaxaar/hynix_2tb_2/Llama-2-13B-GPTQ"
# MODEL = "/run/media/ixaxaar/hynix_2tb_2/Wizard-Vicuna-30B-Uncensored-GPTQ"
# MODEL = "/run/media/ixaxaar/hynix_2tb_2/WizardLM-Uncensored-Falcon-40B-GPTQ"
MODEL = "/run/media/ixaxaar/hynix_2tb_2/CodeLlama-34B-Python-GPTQ"

# Test data: list of symptoms and diseases
test_conditions = [
    ["fatigue", "headache", "cough"],
    ["high blood pressure", "chest pain"],
    ["fever", "rash"],
    ["anxiety", "shortness of breath"],
    ["lower back pain", "numbness in arm"],
]


@pytest.fixture(scope="module")
def loaded_model():
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


@pytest.mark.parametrize("conditions", test_conditions)
def test_generate_follow_up_questions(conditions, loaded_model):
    expected = {0: {"conditions": conditions, "follow_up_questions": []}}  # Replace with expected output

    tokenizer, model = loaded_model

    result = generate_follow_up_questions(
        tokenizer,
        model,
        conditions,
        decoding_strategy="generate",
        temperature=0.7,
        do_sample=True,
        max_new_tokens=100,
    )

    # Validate the result
    assert result["conditions"] == conditions
    assert isinstance(result["follow_up_questions"], list)
    assert len(result["follow_up_questions"]) == len(conditions)
