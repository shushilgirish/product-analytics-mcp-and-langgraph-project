# Use Astronomer Runtime as base image
FROM quay.io/astronomer/astro-runtime:12.8.0

# Install system dependencies
USER root

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


USER astro

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --prefer-binary