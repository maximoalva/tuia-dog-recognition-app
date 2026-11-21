from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import cv2
import numpy as np

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

    # El detector se carga una sola vez. Los jobs corren en paralelo (MAX_WORKERS) y el
    # predictor de ultralytics guarda estado interno, por eso la carga y la inferencia se
    # serializan con un lock.
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
            # ultralytics interpreta los np.ndarray como BGR (convencion OpenCV): no se convierte.
            # classes y conf filtran dentro del detector, antes del NMS.
            result = detector.predict(
                image, conf=self.conf_threshold, classes=[self.dog_class_id], verbose=False
            )[0]

        detections: list[tuple[tuple[int, int, int, int], float]] = []
        for xyxy, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            # Coordenadas en pixeles de la imagen original, recortadas a sus bordes.
            box = self._clip_xyxy(*(int(round(float(v))) for v in xyxy), height, width)
            detections.append((box, float(confidence)))
        detections.sort(key=lambda detection: detection[1], reverse=True)
        return detections

    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza del recorte de un perro detectado usando el modelo
        entrenado en la Etapa 2 (self.classifier.load_model()).

        El recorte llega en BGR (OpenCV). Retorna (raza, score).
        """
        raise NotImplementedError("Etapa 3: implementar classify_detected_dog")

    # ------------------------------------------------------------------
    # Helpers privados de la Etapa 3
    # ------------------------------------------------------------------

    def _get_detector(self) -> Any:
        """YOLO preentrenado en COCO (YOLO_MODEL). Se llama con _detector_lock tomado.

        Si el archivo de pesos no existe, ultralytics lo descarga en esa ruta en el primer uso.
        """
        if DetectionService._detector is None:
            # Import lazy: bootstrap importa este modulo y en Colab (2-train.ipynb) ultralytics
            # no esta instalado; solo se necesita al detectar.
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
