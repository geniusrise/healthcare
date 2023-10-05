import ast
import logging
import re
from typing import Any, Dict, List, Union

import torch
from transformers import AutoTokenizer, GenerationMixin

log = logging.getLogger(__name__)


def extract(text: str) -> List[str]:
    """
    Extracts follow-up questions from a given text block.

    Parameters:
    - text (str): The text block containing follow-up questions.

    Returns:
    - List[str]: A list of extracted follow-up questions.
    """
    # Regular expression pattern to match the array of follow-up questions
    pattern = r"Here is an array of [0-9]* follow up questions:\n```python\n\[(.*?)\]\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        question_list_str = match.group(1)
        return list(ast.literal_eval(question_list_str))
    return []


def prompt(conditions: List[str]) -> str:
    """
    Generates a prompt asking for follow-up questions based on symptoms and diseases.

    Parameters:
    - conditions (List[str]): List of conditions or symptoms to generate questions for.

    Returns:
    - str: A formatted prompt string asking for follow-up questions.
    """
    num_conditions = len(conditions)
    _conditions = '["{cond}"]'.format(cond='", "'.join(conditions))

    return """
## Task

Given the list of symptoms and diseases below, your task is to generate a set of follow-up questions to ask the patient.
These questions should help healthcare professionals gather more detailed information for a more accurate pre-clinical analysis.
Please adhere to the following guidelines:

1. Address the patient in the first person.
2. Generate questions that are easy for a patient to understand; avoid medical jargon.
3. Limit the questions to those that are directly relevant to the listed symptoms and diseases.
4. Do not infer or assume any conditions not listed; stick to the provided list.
5. Dont ask questions that the patient has already told you.

List of Symptoms and Diseases:
```python
conditions = {conditions}
```

Here is an array of {num_conditions} follow up questions:
```python
[\"""".format(
        conditions=_conditions, num_conditions=5 if num_conditions > 5 else num_conditions
    )


def generate_follow_up_questions(
    tokenizer: AutoTokenizer,
    model: GenerationMixin,
    data: List[str],
    max_iterations: int = 1024,
    decoding_strategy: str = "generate",
    **generation_params: Any,
) -> Dict[str, List[str]]:
    """
    Generate follow-up questions using a Hugging Face model and tokenizer.

    Parameters:
    - tokenizer (AutoTokenizer): The Hugging Face tokenizer instance.
    - model (GenerationMixin): The Hugging Face model instance that supports text generation.
    - data (List[str]): The list of strings containing symptoms and diseases.
    - max_iterations (int, optional): Maximum number of iterations for text generation. Default is 1024.
    - decoding_strategy (str, optional): The decoding strategy to use for text generation. Default is 'generate'.
      - 'generate': Basic text generation (default). Relevant params: all.
      - 'greedy_search': Greedy search. Relevant params: max_length.
      - 'contrastive_search': Contrastive search. Relevant params: max_length.
      - 'sample': Sampling. Relevant params: do_sample, temperature, top_p, max_length.
      - 'beam_search': Beam search. Relevant params: num_beams, max_length.
      - 'beam_sample': Beam sampling. Relevant params: num_beams, temperature, max_length.
      - 'group_beam_search': Group beam search. Relevant params: num_beams, diversity_penalty, max_length.
      - 'constrained_beam_search': Constrained beam search. Relevant params: num_beams, max_length, constraints.
    - **generation_params (Any): Additional parameters for text generation. These override the defaults.

    Returns:
    - Dict[str, List[str]]: A dictionary containing the conditions and their corresponding generated follow-up questions.
    """

    results: Dict[int, Dict[str, Union[str, List[str]]]] = {}
    eos_token_id = model.config.eos_token_id

    # Default parameters for each strategy
    default_params = {
        "generate": {"max_length": 4096},
        "greedy_search": {"max_length": 4096},
        "contrastive_search": {"max_length": 4096},
        "sample": {"do_sample": True, "temperature": 0.6, "top_p": 0.9, "max_length": 4096},
        "beam_search": {"num_beams": 4, "max_length": 4096},
        "beam_sample": {"num_beams": 4, "temperature": 0.6, "max_length": 4096},
        "group_beam_search": {"num_beams": 4, "diversity_penalty": 0.5, "max_length": 4096},
        "constrained_beam_search": {"num_beams": 4, "max_length": 4096, "constraints": None},
    }

    # Merge default params with user-provided params
    strategy_params = {**default_params.get(decoding_strategy, {}), **generation_params}  # type: ignore

    # Map of decoding strategy to method
    strategy_to_method = {
        "generate": model.generate,
        "greedy_search": model.greedy_search,
        "contrastive_search": model.contrastive_search,
        "sample": model.sample,
        "beam_search": model.beam_search,
        "beam_sample": model.beam_sample,
        "group_beam_search": model.group_beam_search,
        "constrained_beam_search": model.constrained_beam_search,
    }

    try:
        log.info(f"Generating follow-up questions for document {data}")

        prompt_text = prompt(conditions=data)
        inputs = tokenizer(prompt_text, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to("cuda:0")

        input_ids = inputs["input_ids"]

        # Use the specified decoding strategy
        decoding_method = strategy_to_method.get(decoding_strategy, model.generate)
        generated_ids = decoding_method(input_ids, **strategy_params)

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        log.info(f"Generated text: {generated_text}")

        follow_up_questions = extract(generated_text)
        log.warn(f"Generated follow-up questions: {follow_up_questions}")

        return {"conditions": data, "follow_up_questions": follow_up_questions}

    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
