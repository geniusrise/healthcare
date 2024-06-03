#!/usr/bin/env bash

export NIH_API_KEY=""

curl -o umls.zip https://uts-ws.nlm.nih.gov/download?url=https://download.nlm.nih.gov/umls/kss/2024AA/umls-2024AA-metathesaurus-full.zip&apiKey=$NIH_API_KEY
unzip umls.zip

wget https://lhncbc.nlm.nih.gov/semanticnetwork/download/sn_current.tgz
tar -xvf sn_current.tgz
mv 2023AA/* ./2024AA/META/
