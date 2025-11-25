FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./

RUN uv sync

ENV PATH="/app/.venv/bin:${PATH}"

COPY . .

# Default command: start a shell (you can override to run main.py or replay.py)
CMD ["bash"]

