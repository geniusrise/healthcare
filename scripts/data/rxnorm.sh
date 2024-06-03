#!/usr/bin/env bash

export NIH_API_KEY=""

curl -o rxnorm.zip https://uts-ws.nlm.nih.gov/download?url=https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_05062024.zip&apiKey=$NIH_API_KEY
unzip rxnorm.zip
