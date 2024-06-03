#!/usr/bin/env bash

export NIH_API_KEY=""

curl -o umls.zip https://uts-ws.nlm.nih.gov/download?url=https://download.nlm.nih.gov/umls/kss/2024AA/umls-2024AA-metathesaurus-full.zip&apiKey=$NIH_API_KEY
unzip umls.zip
