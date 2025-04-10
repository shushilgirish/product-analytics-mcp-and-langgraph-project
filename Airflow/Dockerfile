# Use Airflow base image
FROM apache/airflow:2.10.4

# Switch to root to install system dependencies
USER root

# Install essential packages, dev libraries, Chrome, poppler-utils (for pdf2image), etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gpg \
    git \
    pkg-config \
    build-essential \
    cmake \
    python3-dev \
    libpq-dev \
    libssl-dev \
    libonig-dev \
    libbz2-dev \
    libzstd-dev \
    liblzma-dev \
    ca-certificates \
    poppler-utils \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Create directory if you're using ChromaDB
RUN mkdir -p /opt/airflow/chroma_db && chown -R airflow:root /opt/airflow/chroma_db

# Switch to airflow user
USER airflow

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Copy your requirements.txt into container
COPY Airflow/requirements.txt /tmp/requirements.txt

# Create a modified requirements file, removing packages we'll install explicitly
RUN grep -v -E "tokenizers|transformers|sentence-transformers|huggingface_hub" /tmp/requirements.txt > /tmp/filtered_requirements.txt

# Install tokenizers with binary-only flag to avoid compilation
RUN pip install --only-binary=:all: tokenizers==0.15.2

# Install compatible versions of huggingface_hub, transformers and sentence-transformers
# These versions are known to work together
RUN pip install --no-cache-dir huggingface_hub==0.20.3
RUN pip install --no-cache-dir transformers==4.36.2 sentence-transformers==2.2.2

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r /tmp/filtered_requirements.txt --prefer-binary

# (Airflow runs as CMD by default, so no extra command needed)