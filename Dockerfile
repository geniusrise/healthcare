# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev &&
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Create a non-root user and switch to it
RUN useradd -m genius
USER genius

# Make data directories
RUN mkdir -p /app/data/umls /app/data/snomed /app/data/rxnorm /app/data/mesh /app/data/gene_ontology /app/data/disease_ontology

# Run the application
CMD ["uvicorn", "geniusrise_healthcare.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000
