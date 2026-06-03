#!/usr/bin/env bash
# Proyecto 5 setup — RunPod (imagen con PyTorch + CUDA ya instalados).
# Instala lo que falta para el U-Net de segmentación.
set -e

pip install --quiet segmentation-models-pytorch opencv-python-headless

echo "=== Deps instaladas. Datos esperados en 05_understanding_clouds/data/ ==="
echo "   (train.csv, train_images/, test_images/ — descargar de Kaggle, gitignored)"
echo ""
echo "Entrenar:   python 05_understanding_clouds/pipeline.py --train"
echo "Predecir:   python 05_understanding_clouds/pipeline.py --predict"
