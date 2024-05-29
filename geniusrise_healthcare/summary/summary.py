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

import logging
import re
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, GenerationMixin

log = logging.getLogger(__name__)


def extract1(text: str) -> Optional[str]:
    """
    Extracts the summary report from a given text block.

    Parameters:
    - text (str): The text block containing the summary report.

    Returns:
    - Optional[str]: The extracted summary report as a string, or None if not found.
    """
    # Regular expression pattern to match the summary report
    pattern = r"Here is the report in less than 500 words:\n\n```markdown\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def prompt1(conditions: List[str], qa: Dict[str, str], symptoms_diseases: List[str]) -> str:
    """
    Generates a prompt for creating a medical in-patient report for a doctor's consumption.

    Parameters:
    - conditions (List[str]): List of conditions or symptoms mentioned by the patient.
    - qa (Dict[str, str]): Dictionary of questions asked and answers provided by the patient.

    Returns:
    - str: A formatted prompt string for generating the medical report.
    """
    cond = "\n".join([f"- {x}" for x in conditions])
    _qa = "\n\n".join([f"- **Question**: {q} \n  - **Answer**: {a}" for q, a in qa.items()])

    return f"""
## Task

Generate a comprehensive medical in-patient report for a doctor's review.
The patient has presented with the following conditions: {cond}.
Further questions were asked to understand the symptoms better.

Please generate a summary report in less than 500 words in markdown format that includes:

- the patient's observations
- summarizes the questions that were asked
- recommends tests to be conducted and diseases to be checked to further narrow down the cause

Here is the report structure:

```
# In-Patient Report

## Summary

## Recommended tests

### Tests for diagnosis

### Tests for exclusion

## Diseases to check for and narrow down the cause
```

Here is the report in less than 500 words:

```markdown
# In-Patient Report

## Summary

### Patient's Observations

The patient presented with the following complaints:

{symptoms_diseases}

of which these are the conditions identified from SNOMED:

{cond}

### Questions and Answers

Subsequently, on further questioning, the patient had these to add on:

{_qa}

## Recommended tests

### Tests for diagnosis

""".format(
        cond=cond, _qa=_qa, symptoms_diseases=", ".join(symptoms_diseases)
    )


def extract2(text: str) -> Optional[str]:
    """
    Extracts the summary report from a given text block.

    Parameters:
    - text (str): The text block containing the summary report.

    Returns:
    - Optional[str]: The extracted summary report as a string, or None if not found.
    """
    # Regular expression pattern to match the summary report
    pattern = r"The patient should visit the following department:\n\n```markdown\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def prompt2(conditions: List[str], qa: Dict[str, str], symptoms_diseases: List[str]) -> str:
    """
    Generates a prompt asking for follow-up questions based on symptoms and diseases.

    Parameters:
    - **kwargs (Any): Keyword arguments that can be used to format the prompt string.

    Returns:
    - str: A formatted prompt string asking for follow-up questions.
    """
    cond = ", ".join(symptoms_diseases)
    _qa = "\n\n".join([f"Question: {q}? \n Answer: {a}" for q, a in qa.items()])

    return """
## Task

We need to recommend the patient to visit a department or consult with a doctor of a speciality.
If the condition is generic, we need to recommend a general physician.

Example:

```
### Department

The patient should visit the **Outpatient** department.
```

The patient presented with the conditions {conditions}
On further enquiry, a set of questions were asked to the patient, here are the questions and their answers:

{qa}

The patient should visit the following department:

```markdown
### Department

The patient should visit the **""".format(
        conditions=cond, qa=_qa
    )


def generate_summary(
    tokenizer: AutoTokenizer,
    model: GenerationMixin,
    conditions: List[str],
    qa: Dict[str, str],
    symptoms_diseases: List[str],
    max_iterations: int = 1024,
    decoding_strategy: str = "generate",
    **generation_params: Any,
) -> dict:
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
        log.info(f"Generating summary for questions {qa}")

        # 1: generate summary report for doctor
        prompt_text = prompt1(conditions=conditions, qa=qa, symptoms_diseases=symptoms_diseases)
        inputs = tokenizer(prompt_text, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to("cuda:0")

        input_ids = inputs["input_ids"]

        # Use the specified decoding strategy
        decoding_method = strategy_to_method.get(decoding_strategy, model.generate)
        generated_ids = decoding_method(input_ids, **strategy_params)

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        log.info(f"Generated text: {generated_text}")

        summary = extract1(generated_text)
        log.warn(f"Generated summary: {summary}")

        # 1: generate guidance for patient
        prompt_text = prompt2(conditions=conditions, qa=qa, symptoms_diseases=symptoms_diseases)
        inputs = tokenizer(prompt_text, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to("cuda:0")

        input_ids = inputs["input_ids"]

        # Use the specified decoding strategy
        decoding_method = strategy_to_method.get(decoding_strategy, model.generate)
        if "max_new_tokens" in strategy_params:
            strategy_params["max_new_tokens"] = 10
        generated_ids = decoding_method(input_ids, **strategy_params)

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        log.info(f"Generated text: {generated_text}")

        speciality = extract2(generated_text)
        log.warn(f"Generated speciality and department: {speciality}")

        return {"conditions": conditions, "qa": qa, "summary": summary, "speciality": speciality}

    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
