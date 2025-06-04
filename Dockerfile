FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirror.katapult.io/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirror.katapult.io/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y curl git cmake build-essential gcc git python3-opencv && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY . .

RUN uv sync --python 3.12

CMD ["uv", "run", "serve", "run", "ocr_onnx.yaml"]