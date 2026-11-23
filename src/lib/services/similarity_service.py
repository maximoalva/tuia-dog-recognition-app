from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
from torchvision import models

from lib.preprocessing import preprocess
from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)


class SimilarityService:
    """Etapa 1: buscador de imagenes por similitud.

    Funciones a implementar por el estudiante:
      - extract_embedding(image)
      - search_similar_images(embedding, top_k)
      - predict_breed_from_neighbors(results)

    La orquestacion (search, index_image, persistencia y metricas de similitud)
    ya esta provista y no debe modificarse sin justificarlo en el Pull Request.
    """

    # Se carga una sola vez. El lock evita cargarlo dos veces con requests en paralelo.
    _backbone: Optional[torch.nn.Module] = None
    _backbone_lock = threading.Lock()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        top_k: int,
        image_size: int,
        model_name: str,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
        data_root: Optional[Path] = None,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.image_size = image_size
        self.model_name = model_name
        self.url_resolver = url_resolver
        # DATA_PATH: las rutas de la galeria se guardan relativas a esta carpeta para que
        # la base indexada desde el entorno local sirva tambien en Docker (y viceversa).
        self.data_root = data_root

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV)
        return image

    # ------------------------------------------------------------------
    # Etapa 1: funciones a implementar
    # ------------------------------------------------------------------

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando un modelo pre-entrenado en
        ImageNet (ej: ResNet50, EfficientNet, ConvNeXt) sin la capa de
        clasificacion final.

        Sugerencias:
          - Preprocesar la imagen (resize a self.image_size, normalizacion ImageNet).
          - Usar torchvision.models o timm con pesos pre-entrenados.
          - Recordar que la imagen llega en BGR (OpenCV).
        Retorna una lista de floats de dimension EMBEDDING_DIM.
        """
        model = self._get_backbone()
        # Mismo preprocesamiento que en el entrenamiento.
        batch = preprocess(image, self.image_size).unsqueeze(0).to(self._device)
        with torch.no_grad():
            embedding = model(batch)[0].cpu().numpy()
        # Normalizacion L2, asi el producto escalar es la similitud coseno.
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(float).tolist()

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """
        Recupera de la base vectorial las top_k imagenes mas similares.

        Sugerencias:
          - Con pgvector: self.store.search(embedding, top_k).
          - Con JSON: iterar self.store.all() y usar self.similarity(...).
          - Respetar SIMILARITY_METRIC (cosine | l2).
        Retorna una lista de Neighbor (path, breed, score) ordenada por score
        descendente.
        """
        if top_k <= 0:
            return []
        store_search = getattr(self.store, "search", None)
        if callable(store_search) and self.similarity_metric.lower() == "cosine":
            # pgvector resuelve el top-k en la base.
            candidates = store_search(embedding, top_k)
        else:
            # store.search solo ordena por coseno, con l2 (o con JSON) se recorre todo.
            candidates = self.store.all()

        neighbors = [
            Neighbor(path=record.path, breed=record.breed, score=self.similarity(embedding, record.embedding))
            for record in candidates
        ]
        neighbors.sort(key=lambda neighbor: neighbor.score, reverse=True)
        return neighbors[:top_k]

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Predice la raza a partir de los vecinos recuperados (ej: voto
        mayoritario, opcionalmente ponderado por score).

        Si el mejor score esta por debajo de self.similarity_threshold se
        considera "unknown". Retorna (raza, score).

        Voto ponderado, cada vecino suma su score a su raza. En empate gana la del vecino mas
        parecido. Devuelve el score medio de la raza ganadora.
        """
        if not results:
            return "unknown", 0.0
        best_score = max(neighbor.score for neighbor in results)
        if best_score < self.similarity_threshold:
            return "unknown", best_score

        scores_by_breed: dict[str, list[float]] = defaultdict(list)
        for neighbor in results:
            scores_by_breed[neighbor.breed].append(neighbor.score)
        breed, scores = max(scores_by_breed.items(), key=lambda item: (sum(item[1]), max(item[1])))
        return breed, float(np.mean(scores))

    # ------------------------------------------------------------------
    # Helpers privados de la Etapa 1
    # ------------------------------------------------------------------

    def _get_backbone(self) -> torch.nn.Module:
        """ResNet18 preentrenada en ImageNet sin la capa de clasificacion (embedding de 512).

        Es la misma red que el Modelo A y es mas liviana que ResNet50.
        """
        if SimilarityService._backbone is None:
            with SimilarityService._backbone_lock:
                if SimilarityService._backbone is None:
                    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                    model.fc = torch.nn.Identity()
                    model.eval().to(self._device)
                    SimilarityService._backbone = model
                    logger.info("Baseline ResNet18 (ImageNet) cargado en %s", self._device)
        return SimilarityService._backbone

    # ------------------------------------------------------------------
    # Helpers de similitud provistos
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def _to_stored_path(self, image_path: str) -> str:
        """Ruta relativa a DATA_PATH si la imagen esta dentro; si no, la ruta tal cual."""
        if self.data_root is None:
            return str(image_path)
        try:
            return Path(image_path).resolve().relative_to(self.data_root.resolve()).as_posix()
        except ValueError:
            return str(image_path)

    def resolve_path(self, stored_path: str) -> Path:
        """Inversa de _to_stored_path: ruta local de una imagen guardada en la base."""
        path = Path(stored_path)
        if self.data_root is None or path.is_absolute():
            return path
        candidate = self.data_root / path
        if not candidate.exists() and path.exists():
            # Base indexada antes de guardar rutas relativas a DATA_PATH.
            return path
        return candidate

    def index_image(
        self, image_path: str, breed: str, metadata: dict[str, object] | None = None
    ) -> EmbeddingRecord:
        """Extrae el embedding de una imagen del dataset y lo persiste en la base vectorial.

        La ruta se guarda relativa a DATA_PATH (ej: dataset/train/<raza>/<imagen>.jpg).
        Para leer la imagen a partir de un registro, usar resolve_path(record.path).
        """
        image = self._load_image(image_path)
        embedding = self.extract_embedding(image)
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=self._to_stored_path(image_path),
            breed=breed,
            metadata=metadata or {},
        )
        self.store.append(record)
        return record

    def _with_url(self, neighbor: Neighbor) -> Neighbor:
        if self.url_resolver is not None and not neighbor.url:
            neighbor.url = self.url_resolver(self.resolve_path(neighbor.path))
        return neighbor

    def search(
        self,
        source_path: str,
        output_path: Path,
        embedding_fn: Optional[Callable[[np.ndarray], list[float]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Pipeline completo de la Etapa 1: embedding -> vecinos -> raza predicha.

        `embedding_fn` permite seleccionar dinamicamente el extractor
        (baseline, resnet18_finetuned o cnn_custom, ver Etapa 2).
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        extractor = embedding_fn or self.extract_embedding
        embedding = extractor(image)

        k = int(top_k) if top_k else self.top_k
        neighbors = [self._with_url(n) for n in self.search_similar_images(embedding, k)]
        breed, score = self.predict_breed_from_neighbors(neighbors)
        logger.info("Predicted breed: %s (score=%.4f) for %s", breed, score, source_path)

        payload = SearchResult(
            source_path=source_path,
            model=model_name or self.model_name,
            predicted_breed=breed,
            score=round(float(score), 4),
            neighbors=neighbors,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
