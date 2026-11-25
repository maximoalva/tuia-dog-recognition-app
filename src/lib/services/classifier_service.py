from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime

from lib.preprocessing import preprocess
from lib.training import TRAIN_CONFIGS, classification_metrics, embed, load_splits, predict, train_model

logger = logging.getLogger(__name__)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 (o ResNet50) pre-entrenado.
        Modelo B (obligatorio): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en 1-pre-processing.ipynb.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).

        Los hiperparametros salen de TRAIN_CONFIGS (lib/training.py). El historial por epoca
        queda en output/history_<checkpoint>.json para graficar las curvas en la notebook.
        """
        config = TRAIN_CONFIGS[self.active_model_name]
        checkpoint = Path(config.checkpoint) if config.checkpoint else self.active_checkpoint
        history = train_model(self.active_model_name, config, self.dataset_path, self.image_size, checkpoint)

        self.output_path.mkdir(parents=True, exist_ok=True)
        record = {"model": self.active_model_name, "checkpoint": str(checkpoint),
                  "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
                  "history": history}
        history_file = self.output_path / f"history_{checkpoint.stem}.json"
        history_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
        # El checkpoint cambio, la proxima carga tiene que leer el nuevo.
        self._loaded.pop(self.active_model_name, None)
        logger.info("Se guardaron %s y %s", checkpoint.name, history_file.name)

    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Debe reportar: accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. La matriz de confusion y las
        curvas de entrenamiento se documentan en 2-train.ipynb.

        Retorna un dict con las metricas, ej:
          {"accuracy": 0.91, "precision": 0.90, "recall": 0.89,
           "specificity": 0.99, "f1": 0.90}

        Precision, recall, specificity y F1 son macro, se calculan por raza (one-vs-rest)
        y se promedian, asi cada raza pesa lo mismo.
        """
        model = self.load_model()
        splits, classes = load_splits(self.dataset_path)
        test = splits[splits["split"] == "test"]
        logits = predict(model, self.dataset_path, test, self.image_size)
        metrics = classification_metrics(test["label"].to_numpy(), logits.argmax(1), len(classes))
        logger.info("Test de %s: %s", self.active_model_name, {k: round(v, 4) for k, v in metrics.items()})
        return metrics

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.
        """
        model = self.load_model()
        batch = preprocess(image, self.image_size).unsqueeze(0)
        with torch.no_grad():
            embedding = embed(model.eval(), batch)[0].numpy()
        # Normalizacion L2, igual que el baseline.
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(float).tolist()
