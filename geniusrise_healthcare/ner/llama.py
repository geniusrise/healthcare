import logging
import re
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from transformers import AutoTokenizer, GenerationMixin

log = logging.getLogger(__name__)


def extract(text: str) -> List[str]:
    """
    Extracts SNOMED-CT concept IDs from a given text block.

    Parameters:
    - text (str): The text block containing SNOMED-CT concept IDs.

    Returns:
    - List[str]: A list of extracted SNOMED-CT concept IDs.
    """
    # Regular expression pattern to match SNOMED-CT IDs
    pattern = r"\[([^\]]+)\]"
    match = re.search(pattern, text)
    if match:
        snomed_list_str = match.group(1)
        snomed_list = snomed_list_str.split(", ")
        snomed_list = [x.replace('"', "").replace("'", "") for x in snomed_list]  # type: ignore
        return snomed_list
    return []


def prompt(**kwargs: Any) -> str:
    """
    Generates a prompt asking to extract SNOMED-CT concept IDs from a given sentence.

    Parameters:
    - **kwargs (Any): Keyword arguments that can be used to format the prompt string.

    Returns:
    - str: A formatted prompt string asking to extract SNOMED-CT concept IDs.

    Usage:
    >>> prompt(input="311931004 736545008 440547007 112687008 440547006")
    'can you take this sentence and extract all the SNOMED-CT concepts from it: 311931004 736545008 440547007 112687008 440547006'
    """
    return """
## Task

Given the user input below, identify the top 3 specific symptoms and diseases mentioned. Please note the following guidelines:

- Each term may contain multiple words.
- Limit the output to only the top 3 most relevant terms.
- Dont include terms not part of the input.
- Dont infer anything, just perform named entity recognition.
- Avoid generic terms like "symptom" or "disease".
- Stop output after the code block is complete, dont output anything else apart from the 3 symptoms and diseases.

- The user input is as follows:

```python
input = "{input}"
```

Here is the requested array of at max 3 symptoms and diseases referred by the patient:

```python
[\"""".format(
        **kwargs
    )


def annotate_snomed(
    tokenizer: AutoTokenizer,
    model: GenerationMixin,
    data: pd.DataFrame,
    type_ids_filter: List[str],
    page_size: int = 1024,
    max_iterations: int = 1024,
    decoding_strategy: str = "generate",
    **generation_params: Any,
) -> Dict[int, Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]]:
    """
    Annotate text with SNOMED concept IDs using a Hugging Face model and tokenizer.

    Parameters:
    - tokenizer (AutoTokenizer): The Hugging Face tokenizer instance.
    - model (GenerationMixin): The Hugging Face model instance that supports text generation.
    - data (pd.DataFrame): The DataFrame containing text to be annotated.
    - type_ids_filter (List[str]): List of type IDs to filter annotations.
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
      - 'bos_token_id' (int): Beginning of sentence token. Default is 1.
      - 'do_sample' (bool): Whether to sample the next token. Default is True.
      - 'eos_token_id' (int): End of sentence token. Default is 2.
      - 'pad_token_id' (int): Padding token. Default is 0.
      - 'temperature' (float): Controls randomness. Default is 0.6.
      - 'max_length' (int): Maximum length of generated text. Default is 4096.
      - 'top_p' (float): Controls nucleus sampling. Default is 0.9.

    Returns:
    - Dict[int, Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]]: A dictionary containing the document ID and its annotations.

    Usage:
    ```python
    tokenizer, model = load_huggingface_model("model_name")
    data = pd.read_csv("path/to/data.csv")
    type_ids_filter = ['T047', 'T048']
    results = annotate_snomed(tokenizer, model, data, type_ids_filter, decoding_strategy='beam_search', temperature=0.7)
    ```

    """
    # Initialize results dictionary
    results: Dict[int, Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]] = {}
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
        for doc_id, row in data.iterrows():
            text = str(row["text"])
            annotations: List[Dict[str, Union[str, int]]] = []

            for i in range(0, len(text), page_size):
                chunk = text[i : i + page_size]
                log.info(f"Annotating chunk {i // page_size + 1} of document {doc_id}")

                prompt_text = prompt(input=chunk)
                inputs = tokenizer(prompt_text, return_tensors="pt")

                if torch.cuda.is_available():
                    inputs = inputs.to("cuda:0")

                input_ids = inputs["input_ids"]

                # Use the specified decoding strategy
                decoding_method = strategy_to_method.get(decoding_strategy, model.generate)
                generated_ids = decoding_method(input_ids, **strategy_params)

                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                log.info(f"Generated text: {generated_text}")

                extracted_snomed = extract(generated_text)
                log.warn(f"Extracted SNOMED terms: {extracted_snomed}")
                annotations += [
                    {
                        "start": -1,
                        "end": -1,
                        "text": generated_text,
                        "cui": -1,
                        "type_id": -1,
                        "types": -1,
                        "snomed": term,
                    }
                    for term in extracted_snomed
                ]

            results[doc_id] = {"text": text, "annotations": annotations}

        log.info("Annotation process completed.")
        return results

    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
