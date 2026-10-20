FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tmux asciinema && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 pandas==2.2.3 scikit-learn==1.5.2 matplotlib==3.9.2 seaborn==0.13.2 biopython==1.84 pypdf==5.0.1
WORKDIR /workspace
