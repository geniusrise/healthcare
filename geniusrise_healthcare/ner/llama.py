import re
from typing import List, Any


def extract(text: str) -> List[str]:
    """
    Extracts SNOMED-CT concept IDs from a given text block.

    Parameters:
    - text (str): The text block containing SNOMED-CT concept IDs.

    Returns:
    - List[str]: A list of extracted SNOMED-CT concept IDs.
    """
    # Regular expression pattern to match SNOMED-CT IDs
    pattern = r"\b\d{9}\b"

    # Find all matches using the pattern
    snomed_ct_ids = re.findall(pattern, text)

    return snomed_ct_ids


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
can you take this sentence and extract all the SNOMED-CT concepts from it:

```
{input}
```

please output a list of concept ids in a code block, nothing else

""".format(
        **kwargs
    )
