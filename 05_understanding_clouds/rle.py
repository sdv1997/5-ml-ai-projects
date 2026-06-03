"""
Proyecto 5 — Understanding Clouds: utilidades RLE + métrica Dice
================================================================

Convención de la competición (Kaggle):
  - Las máscaras se codifican en RLE *column-major* (orden Fortran): los píxeles
    se numeran de arriba abajo y luego de izquierda a derecha, 1-indexados.
  - La submission es un CSV `Image_Label,EncodedPixels` (una fila por imagen×clase).
  - Métrica: mean Dice sobre cada par <imagen, clase>. Dice = 1 si pred y gt están
    ambos vacíos.

Módulo ligero (solo numpy) para que lo importen tanto eda.py (local) como
pipeline.py (en el pod).
"""
import numpy as np

CLASSES = ["Fish", "Flower", "Gravel", "Sugar"]
IMG_SHAPE = (1400, 2100)  # (alto, ancho) de las imágenes originales


def rle_decode(rle, shape=IMG_SHAPE):
    """RLE (string) -> máscara binaria uint8 (alto, ancho). '' o NaN -> todo ceros."""
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if isinstance(rle, float) or rle is None or rle == "" or (isinstance(rle, str) and not rle.strip()):
        return mask.reshape(shape, order="F")
    s = np.asarray(rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    for lo, ln in zip(starts, lengths):
        mask[lo:lo + ln] = 1
    return mask.reshape(shape, order="F")


def rle_encode(mask):
    """Máscara binaria (alto, ancho) -> RLE string (column-major). Vacía -> ''."""
    pixels = mask.T.flatten()  # column-major
    if pixels.sum() == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def dice(pred, gt):
    """Dice de una máscara binaria vs ground truth. 1.0 si ambas vacías."""
    pred = pred.astype(bool); gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return 2.0 * inter / (pred.sum() + gt.sum())


def mean_dice(preds, gts):
    """Media de Dice sobre una lista de pares (pred_mask, gt_mask) = la métrica oficial."""
    return float(np.mean([dice(p, g) for p, g in zip(preds, gts)]))


def remove_small(mask, min_size):
    """Post-proceso clave: si la máscara predicha tiene menos de min_size píxeles,
    se descarta (se asume 'clase ausente'). Sube mucho el Dice en esta competición."""
    return mask if mask.sum() >= min_size else np.zeros_like(mask)


if __name__ == "__main__":
    # round-trip sanity check
    m = np.zeros(IMG_SHAPE, dtype=np.uint8)
    m[100:300, 200:500] = 1
    assert np.array_equal(rle_decode(rle_encode(m)), m), "RLE round-trip falla"
    assert dice(m, m) == 1.0 and dice(m, np.zeros_like(m)) == 0.0
    assert dice(np.zeros_like(m), np.zeros_like(m)) == 1.0
    print("rle.py OK — round-trip y Dice correctos")
