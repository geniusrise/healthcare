from .llama import annotate_snomed as annotate_snomed_llama2
from .medcat import annotate_snomed as annotate_snomed_medcat


def annotate_snomed(type: str, *args, **kwargs):
    if type == "medcat":
        return annotate_snomed_medcat(*args, **kwargs)
    elif type == "llama2":
        return annotate_snomed_llama2(*args, **kwargs)
    else:
        raise ValueError(f"This type of annotator is not supported: {type}")
