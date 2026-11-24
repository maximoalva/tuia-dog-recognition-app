from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch

from lib.preprocessing import preprocess
from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService

logger = logging.getLogger(__name__)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones a implementar por el estudiante:
      - detect_dogs(image)
      - classify_detected_dog(crop)

    La orquestacion (predict: deteccion -> recorte -> clasificacion -> JSON)
    ya esta provista.
    """
    # El detector se carga una sola vez. Los jobs corren en paralelo (MAX_WORKERS) y el predictor de ultralytics guarda estado interno,
    # por eso la carga y la inferencia se serializan con un lock.
    _detector: Optional[Any] = None
    _detector_lock = threading.Lock()

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV / ultralytics)
        return image

    # ------------------------------------------------------------------
    # Etapa 3: funciones a implementar
    # ------------------------------------------------------------------

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Detecta todos los perros presentes en la imagen usando un modelo YOLO
        pre-entrenado (ej: YOLOv8n via ultralytics). No es necesario entrenar
        el detector.

        Sugerencias:
          - self.yolo_model_name, self.conf_threshold y self.dog_class_id
            (clase 'dog' = 16 en COCO) vienen de la configuracion (.env).
          - Debe funcionar con un perro, multiples perros y escenas complejas.

        Retorna una lista de ((x1, y1, x2, y2), confidence) en pixeles.
        """
        if image is None or image.ndim != 3 or image.size == 0:
            raise ValueError("detect_dogs espera una imagen BGR (H, W, 3)")
        height, width = image.shape[:2]

        with DetectionService._detector_lock:
            detector = self._get_detector()
            # classes y conf filtran dentro del detector, antes del NMS.
            result = detector.predict(image, conf=self.conf_threshold, classes=[self.dog_class_id], verbose=False)[0]

        detections: list[tuple[tuple[int, int, int, int], float]] = []
        for xyxy, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            # Coordenadas en pixeles de la imagen original, recortadas a sus bordes.
            box = self._clip_xyxy(*(int(round(float(v))) for v in xyxy), height, width)
            detections.append((box, float(confidence)))
        detections.sort(key=lambda detection: detection[1], reverse=True)
        return self._remove_nested_boxes(detections, self.NESTED_BOX_THRESHOLD)

    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza del recorte de un perro detectado usando el modelo
        entrenado en la Etapa 2 (self.classifier.load_model()).

        El recorte llega en BGR (OpenCV). Retorna (raza, score).
        """
        # Mismo preprocesamiento que en el entrenamiento.
        tensor = preprocess(crop, self.classifier.image_size).unsqueeze(0)
        model = self.classifier.load_model()
        logits = self._predict_logits(model, tensor)[0]

        # Softmax estable (se resta el máximo para evitar overflow en exp).
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        index = int(probs.argmax())

        classes = self._get_classes(model)
        if len(classes) != len(probs):
            raise ValueError(
                f"El modelo tiene {len(probs)} salidas pero la lista de razas tiene {len(classes)}"
            )
        return classes[index], float(probs[index])

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    # Una caja se descarta si al menos esta fracción de su área queda dentro de otra caja de perro más confiable.
    NESTED_BOX_THRESHOLD = 0.85

    @staticmethod
    def _remove_nested_boxes(
        detections: list[tuple[tuple[int, int, int, int], float]], threshold: float) -> list[tuple[tuple[int, int, int, int], float]]:
        """Descarta las cajas que quedan casi enteras dentro de otra caja más confiable.

        Recibe las detecciones ordenadas por confianza descendente y conserva ese orden.
        """
        kept: list[tuple[tuple[int, int, int, int], float]] = []
        for box, confidence in detections:
            x1, y1, x2, y2 = box
            area = max(1, (x2 - x1) * (y2 - y1))
            nested = False
            for (kx1, ky1, kx2, ky2), _ in kept:
                inter_w = max(0, min(x2, kx2) - max(x1, kx1))
                inter_h = max(0, min(y2, ky2) - max(y1, ky1))
                if inter_w * inter_h / area >= threshold:
                    nested = True
                    break
            if not nested:
                kept.append((box, confidence))
        return kept

    def _get_classes(self, model: Any) -> list[str]:
        """Lista de razas en el orden de las salidas del modelo activo.

        Se busca, en orden: el atributo ``classes`` del modelo (.pth), la metadata ``classes`` del ONNX
        (JSON) y, si el modelo no la trae, ``data/processed/classes.json`` (generado en
        1-pre-processing.ipynb, sección 9). Se cachea por modelo.
        """
        key = self.classifier.active_model_name
        if key in self._classes_cache:
            return self._classes_cache[key]

        classes = getattr(model, "classes", None)
        if classes is None and hasattr(model, "get_modelmeta"):  # onnxruntime.InferenceSession
            raw = model.get_modelmeta().custom_metadata_map.get("classes")
            classes = json.loads(raw) if raw else None
        if classes is None:
            from lib.config import settings

            classes_path = Path(settings.data_path) / "processed" / "classes.json"
            if not classes_path.exists():
                raise ValueError(
                    f"El modelo '{key}' no trae la lista de razas y no existe {classes_path}"
                )
            classes = json.loads(classes_path.read_text(encoding="utf-8"))
            logger.warning("El modelo '%s' no trae la lista de razas: se usa %s", key, classes_path)

        self._classes_cache[key] = list(classes)
        return self._classes_cache[key]

    def _predict_logits(self, model: Any, tensor: torch.Tensor) -> np.ndarray:
        """Corre el modelo (.pth u .onnx) sobre un batch y devuelve los logits como array (batch, clases)."""
        if hasattr(model, "get_inputs"):  # onnxruntime.InferenceSession
            input_name = model.get_inputs()[0].name
            outputs = model.run(None, {input_name: tensor.numpy()})
            return np.asarray(outputs[0])
        model.eval()
        with torch.no_grad():
            output = model(tensor)
        if isinstance(output, (tuple, list)): # por si el forward devuelve (logits, embedding)
            output = output[0]
        return output.detach().cpu().numpy()

    def _get_detector(self) -> Any:
        """YOLO preentrenado en COCO (YOLO_MODEL). Se llama con _detector_lock tomado.

        Si el archivo de pesos no existe, ultralytics lo descarga en esa ruta en el primer uso.
        """
        if DetectionService._detector is None:
            from ultralytics import YOLO

            DetectionService._detector = YOLO(self.yolo_model_name)
            logger.info("Detector YOLO cargado: %s", self.yolo_model_name)
        return DetectionService._detector

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        """Clasifica la imagen completa con el modelo entrenado (pestaña Etapa 2).

        Reutiliza classify_detected_dog tratando la imagen entera como recorte,
        por lo que requiere la Etapa 2 (modelo entrenado) y classify_detected_dog.
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        # El modelo elegido vale solo para esta clasificacion: se restaura el activo para
        # que /detect siga usando el modelo configurado.
        previous = self.classifier.active_model_name
        if model_name:
            self.classifier.set_active_model(model_name)
        try:
            breed, score = self.classify_detected_dog(image)
        finally:
            self.classifier.set_active_model(previous)
        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        """Flujo completo: deteccion -> bounding boxes -> recortes -> clasificacion.

        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score = self.classify_detected_dog(crop)
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
