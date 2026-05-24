#!/usr/bin/env bash
# Day 3 setup — instala dependencias y descarga modelo en /workspace/ (no en overlay raíz)
# Qwen2.5-3B-Instruct (~6 GB) cabe con margen en el volumen de 20 GB
set -e

export HF_HOME=/workspace/.cache/huggingface

echo "=== Instalando dependencias ==="
pip install vllm rouge-score --quiet

echo "=== Descargando Qwen2.5-3B-Instruct a /workspace/.cache (~6 GB) ==="
python3 - <<'EOF'
import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-3B-Instruct", ignore_patterns=["*.gguf"])
print("Modelo descargado OK")
EOF

echo "=== Setup completado ==="
echo "Ejecutar desde /workspace/30-days-of-ai:"
echo "  HF_HOME=/workspace/.cache/huggingface python3 day_03/day03.py --eval-only"
echo "  HF_HOME=/workspace/.cache/huggingface python3 day_03/day03.py"
