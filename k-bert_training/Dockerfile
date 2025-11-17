# Dockerfile for K-BERT Prediction Pipeline (Job 2) - v5 (Fix NumPy)

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    MAMBA_NO_BANNER=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-lc"]
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
RUN micromamba create -y -n app -c conda-forge python=3.10 && micromamba clean -a -y

# --- [!! 關鍵修正 !!] ---
# 鎖定 NumPy 版本 < 2.0，以解決與 PyTorch 2.1.0 的衝突
RUN micromamba run -n app python -m pip install --no-cache-dir \
    torch==2.1.0 \
    transformers==4.35.2 \
    google-cloud-storage \
    rdflib \
    "numpy<2.0"

COPY kbert_model_output_multilabel/ /app/kbert_model_output_multilabel/

COPY ttl_to_json_converter_6.py /app/ttl_to_json_converter_6.py
COPY json_risk_detector_7.py /app/json_risk_detector_7.py
COPY run_kbert_pipeline.py /app/run_kbert_pipeline.py

ENTRYPOINT ["micromamba","run","-n","app","python","/app/run_kbert_pipeline.py"]