import pandas as pd
import pytest
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.config import Config
from pathlib import Path
from spacy.util import load_model_from_path
from typing import List

from geniusrise_healthcare.ner import annotate_text_with_snomed

# Load the CDB and vocab
cdb = CDB.load("models/medcat-snomed/cdb.dat")

# Load the custom SpaCy model
spacy_model_path = Path("models/medcat-snomed/spacy_model")
spacy_model = load_model_from_path(spacy_model_path)

# Update the MedCAT configuration
config = Config()
config.general["spacy_model"] = "models/medcat-snomed/spacy_model"

# Initialize CAT
cat = CAT(cdb, config=config)

# Type IDs to filter
# T184: Symptom
# T047: Disease or Syndrome
# T048: Mental or Behavioral Dysfunction
# T046: Pathological Function
# T121: Pharmacologic Substanc
type_ids_filter: List[str] = []

# Real-world, messy queries from health forums
queries = [
    "I've been feeling extremely fatigued for the past two months. I can barely get out of bed and I'm wondering if this could be a symptom of diabetes or some other underlying condition?",
    "My mother was recently diagnosed with stage 3 ovarian cancer. We're all devastated and confused. What does stage 3 mean in terms of prognosis and treatment options?",
    "I've been experiencing a sharp, stabbing pain in my chest that comes and goes. It's been happening for a week now. Should I be concerned about heart issues?",
    "I've had constant headaches for about three weeks now. Sometimes they're so bad that I can't even focus on work. Could this be a migraine or is it something more serious?",
    "I've noticed that I feel short of breath whenever I climb stairs or walk a short distance. This has been going on for a couple of weeks. Could this be a sign of a heart or lung problem?",
    "For the past few days, I've been feeling a strange numbness in my left arm and hand. It comes and goes but it's starting to worry me. Is this something I should get checked out?",
    "I've had a persistent cough for about two weeks and it's not getting better. With the current COVID-19 situation, I'm really worried. What should I do?",
    "I was recently diagnosed with high blood pressure. I want to manage it through diet if possible. What foods should I avoid to lower my blood pressure?",
    "My 4-year-old has had a high fever for three days and now a rash is starting to appear. I'm really worried it could be measles or another serious illness. What should I do?",
    "I've been experiencing severe lower back pain for about a month now. Sometimes it's so bad that I can't move. Could this be a slipped disc or some other serious condition?",
    "I've been feeling dizzy almost every day for the past two weeks. Sometimes it's so bad that I feel like I'm going to faint. Could this be a neurological issue?",
    "I've had a sore throat for a while now and it's becoming increasingly difficult to swallow. I also have a feeling of a lump in my throat. What could this be?",
    "I've noticed that I've been urinating much more frequently than usual for the past week or so. It's starting to interfere with my daily activities. What could be causing this?",
    "I have a persistent cough that's been keeping me up at night for the past two weeks. I've tried over-the-counter cough medicine but it's not helping. What should I do?",
    "I've been feeling anxious all the time for the past month. I'm constantly worried about everything and it's affecting my quality of life. Could this be an anxiety disorder?",
]

# Test data
test_data = pd.DataFrame({"text": queries})


# Test cases
@pytest.mark.parametrize(
    "data, expected",
    [
        (
            test_data,
            {i: {"text": query, "annotations": ["lol"]} for i, query in enumerate(queries)},
        ),  # Real-world queries
        # (pd.DataFrame({"text": ["Short text"]}), {0: {"text": "Short text", "annotations": []}}),  # Single short text
        # (pd.DataFrame({"text": [""]}), {0: {"text": "", "annotations": []}}),  # Empty text
    ],
)
def test_annotate_text_with_snomed(data, expected):
    result = annotate_text_with_snomed(cat, data, type_ids_filter)
    assert result == expected
