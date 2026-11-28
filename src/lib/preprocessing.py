"""Preprocesamiento de imágenes.
load_bgr -> bgr_to_rgb -> [augmentation, solo en train] -> resize_to_square -> normalize
preprocess() encadena todos los pasos y es la función que usan los servicios (augment=False, por defecto)
y el entrenamiento (augment=True).
"""
from __future__ import annotations

import logging
from pathlib import Path
from functools import lru_cache
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import torch

from lib.augmentation import build_train_augmentation

logger = logging.getLogger(__name__)

# Estadísticas de ImageNet (RGB, escala 0-1): las que usaron los modelos preentrenados de torchvision.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Color de relleno del padding (RGB, 0-255): la media de ImageNet, que después de normalizar vale 0.
PAD_COLOR = tuple(round(m * 255) for m in IMAGENET_MEAN)

ResizeMode = Literal["pad", "stretch"]
DEFAULT_RESIZE_MODE: ResizeMode = "pad"


def load_bgr(path: str | Path) -> np.ndarray:
    """
    Lee una imagen del disco en BGR uint8, como ``cv2.imread``, pero fallando con un mensaje claro.

    ``cv2.imread`` devuelve ``None`` sin avisar si el archivo no existe o no se puede decodificar, y no
    funciona en Windows con rutas que tienen acentos o ñ. Leer los bytes con NumPy y decodificarlos con
    ``cv2.imdecode`` evita los dos problemas.

    Args:
        path (str | Path): Ruta a la imagen.

    Returns:
        np.ndarray: Imagen BGR uint8 de forma (alto, ancho, 3).

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo no se puede decodificar como imagen.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo decodificar la imagen: {path}")
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen BGR (OpenCV) a RGB (PIL, torchvision, albumentations y modelos preentrenados).

    Args:
        image (np.ndarray): Imagen BGR uint8 de forma (alto, ancho, 3).

    Returns:
        np.ndarray: Imagen RGB uint8 de forma (alto, ancho, 3).
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_to_square(image: np.ndarray, size: int, mode: ResizeMode = DEFAULT_RESIZE_MODE) -> np.ndarray:
    """
    Lleva una imagen de cualquier relación de aspecto a ``size`` x ``size``.

    - ``"pad"``: achica o agranda la imagen respetando su relación de aspecto y rellena los bordes con
      :data:`PAD_COLOR`. No deforma al perro (su forma es parte de la raza) y no recorta ninguna parte.
    - ``"stretch"``: resize directo, que deforma la imagen si no es cuadrada. Se mantiene para comparar.

    Se usa ``INTER_AREA`` para achicar (evita el aliasing) e ``INTER_LINEAR`` para agrandar, por ejemplo
    con recortes chicos de YOLO.

    Args:
        image (np.ndarray): Imagen RGB uint8 de forma (alto, ancho, 3).
        size (int): Lado de la imagen de salida, en píxeles (``IMAGE_SIZE``).
        mode (ResizeMode): ``"pad"`` (por defecto) o ``"stretch"``.

    Returns:
        np.ndarray: Imagen RGB uint8 de forma (size, size, 3).
    """
    h, w = image.shape[:2]
    if mode == "stretch":
        interpolation = cv2.INTER_AREA if h * w > size * size else cv2.INTER_LINEAR
        return cv2.resize(image, (size, size), interpolation=interpolation)
    if mode != "pad":
        raise ValueError(f"Modo de resize desconocido: {mode!r} (usar 'pad' o 'stretch')")

    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = PAD_COLOR
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


@lru_cache(maxsize=None)
def train_augmentation(size: int, seed: int | None = None) -> A.Compose:
    """
    Pipeline de data augmentation para el entrenamiento, definido en :mod:`lib.augmentation`.

    Se construye una sola vez por combinación de ``size`` y ``seed`` (``lru_cache``) y se reutiliza en cada
    llamada a :func:`preprocess`: si no, se volvería a armar para cada imagen de cada época.

    Args:
        size (int): Lado de la imagen de salida, en píxeles (``IMAGE_SIZE``).
        seed (int | None): Semilla del pipeline, para que el entrenamiento sea reproducible.

    Returns:
        A.Compose: Pipeline de albumentations.
    """
    return build_train_augmentation(size, seed=seed)


def normalize(image: np.ndarray) -> torch.Tensor:
    """
    Normaliza una imagen RGB uint8 con las estadísticas de ImageNet y la convierte en tensor CHW.

    Args:
        image (np.ndarray): Imagen RGB uint8 de forma (alto, ancho, 3).

    Returns:
        torch.Tensor: Tensor float32 de forma (3, alto, ancho).
    """
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = (image.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def preprocess(image: np.ndarray, size: int, mode: ResizeMode = DEFAULT_RESIZE_MODE, augment: bool = False, seed: int | None = None) -> torch.Tensor:
    """
    Preprocesamiento completo: BGR -> RGB -> [augmentation] -> resize a ``size`` x ``size`` -> normalización.

    Es la función que usan los servicios de la app, la evaluación y el entrenamiento. El augmentation se
    aplica antes del resize, sobre la imagen original: así, por ejemplo, una rotación gira la foto y no las
    bandas del padding. Devuelve un tensor sin dimensión de batch: para pasarlo al modelo, usar
    ``preprocess(...).unsqueeze(0)``.

    Args:
        image (np.ndarray): Imagen BGR uint8 (como ``cv2.imread``), de forma (alto, ancho, 3).
        size (int): Lado de la imagen de salida, en píxeles (``IMAGE_SIZE``).
        mode (ResizeMode): ``"pad"`` (por defecto) o ``"stretch"``.
        augment (bool): Si es ``True``, aplica :func:`train_augmentation`. Solo para imágenes de ``train``:
            nunca en validación, test, el conjunto independiente ni la app.
        seed (int | None): Semilla del augmentation (solo se usa con ``augment=True``).

    Returns:
        torch.Tensor: Tensor float32 de forma (3, size, size).

    Raises:
        ValueError: Si la imagen está vacía (por ejemplo, un recorte degenerado) o no tiene 3 canales.
    """
    if image is None or image.size == 0:
        raise ValueError("La imagen está vacía (¿recorte de tamaño 0?)")
    if image.ndim == 2:  # escala de grises
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # BGRA: se descarta la transparencia
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Se esperaba una imagen BGR de 3 canales, llegó una de forma {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Se esperaba una imagen uint8, llegó {image.dtype}")
    rgb = bgr_to_rgb(image)
    if augment:
        rgb = train_augmentation(size, seed)(image=rgb)["image"]
    return normalize(resize_to_square(rgb, size, mode))


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Inversa de :func:`normalize`, para visualizar un tensor preprocesado (por ejemplo, en las notebooks).

    Args:
        tensor (torch.Tensor): Tensor float32 de forma (3, alto, ancho).

    Returns:
        np.ndarray: Imagen RGB uint8 de forma (alto, ancho, 3).
    """
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0) * std + mean
    return (np.clip(array, 0, 1) * 255).round().astype(np.uint8)