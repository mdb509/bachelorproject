FROM python:3.12.3-slim
# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    make \
    gcc \
    g++ \
    libgmp-dev \
    zlib1g-dev \
    ca-certificates \
    curl \
    bash \
    xz-utils \
    bzip2 \
    pkg-config \
    jq \
    unzip \
    && rm -rf /var/lib/apt/lists/*


# Set Work Directory
WORKDIR /workspace

# Clone Repositories
RUN git clone https://github.com/mdb509/bachelorproject.git && \
    git clone https://github.com/arminbiere/dualiza.git && \
    git clone https://github.com/timpehoerig/master_project.git

# Get Ganak Binary (latest release) to /workspace/ganak-linux-amd64/ganak
RUN set -e; u="$(curl -sL https://api.github.com/repos/meelgroup/ganak/releases/latest | jq -r '.assets[]|select(.name|test("linux.*(x86_64|amd64).*\\.zip$";"i"))|.browser_download_url' | head -n1)"; \
  mkdir -p /tmp/g /workspace/ganak-linux-amd64; curl -Ls "$u" -o /tmp/g.zip; unzip -q /tmp/g.zip -d /tmp/g; \
  cp "$(find /tmp/g -type f -name ganak -print -quit)" /workspace/ganak-linux-amd64/ganak; chmod +x /workspace/ganak-linux-amd64/ganak

ENV PATH="/workspace/ganak-linux-amd64:${PATH}"

# Clone CaDiCaL inside master_project
WORKDIR /workspace/master_project
RUN git clone https://github.com/arminbiere/cadical.git cadical

# Python Dependencies
WORKDIR /workspace/bachelorproject
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Run Bench Script
COPY run_bench.sh /run_bench.sh
RUN chmod +x /run_bench.sh

CMD ["/run_bench.sh"]
