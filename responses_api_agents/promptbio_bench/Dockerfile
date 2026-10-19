FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends bash coreutils samtools tabix bcftools bedtools && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir biopython==1.85 numpy==2.2.6 pandas==2.3.0 scipy==1.15.3 pysam==0.23.3
WORKDIR /workspace
