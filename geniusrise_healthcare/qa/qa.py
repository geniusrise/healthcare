import logging
import re
from typing import Any, Dict, List, Union

import pandas as pd
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
    pattern = r"Here is an array of follow up questions:\n```python\n\[(.*?)\]\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        question_list_str = match.group(1)
        question_list = question_list_str.split(", ")
        question_list = [x.strip('"').strip("'") for x in question_list]
        return question_list
    return []


def prompt(**kwargs: Any) -> str:
    """
    Generates a prompt asking for follow-up questions based on symptoms and diseases.

    Parameters:
    - **kwargs (Any): Keyword arguments that can be used to format the prompt string.

    Returns:
    - str: A formatted prompt string asking for follow-up questions.
    """
    return """
Given the list of symptoms and diseases below, we need to generate a set of follow-up questions.
These questions should be easy for a patient to understand and should aim to gather more detailed information about each symptom or condition. The goal is to assist healthcare professionals in making a more accurate pre-clinical analysis.

List of Symptoms and Diseases:
```python
conditions = {conditions}
```

Please generate a list of five follow-up questions that can be asked to the patient to get more details about each listed symptom or disease. The questions should be clear, concise, and non-medical jargon.

Here is an array of follow up questions:
```python
[\"""".format(
        **kwargs
    )


def generate_follow_up_questions(
    tokenizer: AutoTokenizer,
    model: GenerationMixin,
    data: List[str],
    page_size: int = 1024,
    max_iterations: int = 1024,
    decoding_strategy: str = "generate",
    **generation_params: Any,
) -> Dict[str, Union[str, List[str]]]:
    """
    Generate follow-up questions using a Hugging Face model and tokenizer.

    Parameters:
    - tokenizer (AutoTokenizer): The Hugging Face tokenizer instance.
    - model (GenerationMixin): The Hugging Face model instance that supports text generation.
    - data (List[str]): The list of strings containing symptoms and diseases.
    - page_size (int, optional): The size of each text chunk for pagination. Default is 1024 characters.
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
    - Dict[int, Dict[str, Union[str, List[str]]]]: A dictionary containing the document ID and its generated follow-up questions.
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
        conditions = '["{cond}"]'.format(cond='", "'.join(data))
        log.info(f"Generating follow-up questions for document {conditions}")

        prompt_text = prompt(conditions=conditions)
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
        log.info(f"Generated follow-up questions: {follow_up_questions}")

        return {"conditions": conditions, "follow_up_questions": follow_up_questions}

    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
