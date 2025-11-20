# GPU base: CUDA 11.8 + cuDNN 8 + Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    MAMBA_NO_BANNER=1

# 基本系統依賴（OpenCV 需要 libgl1 / libglib2.0-0）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 安裝 micromamba（不會有 Anaconda TOS）
SHELL ["/bin/bash", "-lc"]
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
WORKDIR /app

# 建立 conda-forge 的 Python 3.10 環境
RUN micromamba create -y -n app -c conda-forge python=3.10 && \
    micromamba clean -a -y

# --- 以下是修改的關鍵部分 ---

# 1. 複製 "GPU分析腳本" 用的 requirements
#    (它會複製建立的 requirements_gpu.txt)
COPY requirements_gpu.txt /app/requirements.txt

# 2. 先安裝 PyTorch，再安裝 "GPU分析腳本" 用的 requirements
RUN micromamba run -n app python -m pip install --upgrade pip && \
    micromamba run -n app python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 && \
    micromamba run -n app python -m pip install --no-cache-dir -r /app/requirements.txt

# 3. 複製"GPU分析腳本"
#    (它會複製改名後的 main_gpu.py，並在容器中命名為 main.py)
COPY main_gpu.py /app/main.py

# 4. 以 micromamba 執行腳本
ENTRYPOINT ["micromamba","run","-n","app","python","/app/main.py"]