"""Entrenamiento y evaluacion de los clasificadores de la Etapa 2.

ClassifierService usa este modulo. Los hiperparametros de cada modelo estan en TRAIN_CONFIGS
y se pueden cambiar desde la notebook antes de llamar a train_classifier.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torchvision import models

from lib.evaluation.metrics import precision_recall_f1, specificity
from lib.models.cnn_custom import CNNCustom
from lib.preprocessing import load_bgr, preprocess

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Hiperparametros de un entrenamiento."""

    strategy: str = "finetune"  # ResNet con finetune (toda la red) o frozen (solo la ultima capa), CNN propia con scratch
    augment: bool = True
    epochs: int = 30
    batch_size: int = 64
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    lr_patience: int = 2  # epocas sin mejorar antes de bajar el learning rate
    early_stopping: int = 5  # epocas sin mejorar antes de cortar
    num_workers: int = 4
    seed: int = 42
    checkpoint: Path | None = None  # None guarda en el checkpoint del modelo activo
    # Arquitectura de la CNN propia (solo cnn_custom)
    channels: tuple[int, ...] = (32, 64, 128, 256)
    convs_per_block: int = 1
    embedding_dim: int = 512
    dropout: float = 0.3


TRAIN_CONFIGS: dict[str, TrainConfig] = {
    "resnet18_finetuned": TrainConfig(),
    # Desde cero todas las capas arrancan igual, asi que van con el mismo learning rate.
    "cnn_custom": TrainConfig(strategy="scratch", epochs=50, lr_backbone=1e-3, lr_patience=3, early_stopping=8),
}


def load_splits(dataset_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Splits y lista de razas de la Parte 1 (data/processed)."""
    processed = Path(dataset_path).parent / "processed"
    splits = pd.read_csv(processed / "splits.csv")
    classes = json.loads((processed / "classes.json").read_text(encoding="utf-8"))
    return splits, classes


class BreedDataset(Dataset):
    """Imagenes de un split con su etiqueta, con el mismo preprocesamiento que la app."""

    def __init__(self, dataset_path: Path, df: pd.DataFrame, image_size: int, augment: bool = False) -> None:
        self.paths = [Path(dataset_path) / p for p in df["path"]]
        self.labels = df["label"].to_numpy() if "label" in df else np.zeros(len(df), dtype=int)
        self.image_size = image_size
        self.augment = augment
        self.seed: int | None = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        x = preprocess(load_bgr(self.paths[i]), self.image_size, augment=self.augment, seed=self.seed)
        return x, int(self.labels[i])


def _seed_worker(worker_id: int) -> None:
    # Cada worker y cada epoca recibe una semilla distinta pero reproducible.
    info = get_worker_info()
    seed = info.seed % 2**32
    info.dataset.seed = seed
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset: BreedDataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
        # En Windows cada worker tarda en arrancar, asi se crean una sola vez y no en cada epoca.
        persistent_workers=num_workers > 0,
    )


def build_model(name: str, num_classes: int, config: TrainConfig) -> nn.Module:
    """Modelo nuevo listo para entrenar."""
    if name == "cnn_custom":
        if config.strategy != "scratch":
            raise ValueError("La CNN propia se entrena desde cero (strategy='scratch')")
        return CNNCustom(num_classes, config.channels, config.convs_per_block, config.embedding_dim, config.dropout)
    if name != "resnet18_finetuned":
        raise ValueError(f"No hay receta de entrenamiento para '{name}'")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if config.strategy == "frozen":
        for param_name, param in model.named_parameters():
            param.requires_grad = param_name.startswith("fc.")
    elif config.strategy != "finetune":
        raise ValueError(f"Estrategia desconocida '{config.strategy}' (usar finetune o frozen)")
    return model


def _set_train_mode(model: nn.Module, strategy: str) -> None:
    # Con la red congelada, BatchNorm tiene que seguir usando las estadisticas de ImageNet.
    model.train(strategy != "frozen")


def _run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
               optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    """Loss y accuracy medias de una pasada. Con optimizer tambien entrena."""
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def save_model(model: nn.Module, classes: list[str], path: Path) -> None:
    """Modelo completo en CPU con la lista de razas, el formato que lee ClassifierService.load_model."""
    best = copy.deepcopy(model).cpu().eval()
    best.classes = list(classes)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, path)


def train_model(name: str, config: TrainConfig, dataset_path: Path, image_size: int, checkpoint: Path) -> list[dict]:
    """Entrena con early stopping, guarda el mejor modelo segun valid y devuelve el historial por epoca."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, classes = load_splits(dataset_path)
    train_ds = BreedDataset(dataset_path, splits[splits["split"] == "train"], image_size, augment=config.augment)
    valid_ds = BreedDataset(dataset_path, splits[splits["split"] == "valid"], image_size)
    train_loader = make_loader(train_ds, config.batch_size, True, config.num_workers, config.seed)
    valid_loader = make_loader(valid_ds, config.batch_size, False, config.num_workers, config.seed)

    model = build_model(name, len(classes), config).to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.") and p.requires_grad]
    groups = [{"params": head, "lr": config.lr_head}]
    if backbone:
        groups.append({"params": backbone, "lr": config.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=config.lr_patience)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Entrenando %s (%s) en %s, %d parametros entrenables", name, config.strategy, device, trainable)

    history: list[dict] = []
    best_loss, wait = math.inf, 0
    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        _set_train_mode(model, config.strategy)
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        model.eval()
        valid_loss, valid_acc = _run_epoch(model, valid_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(valid_loss)

        improved = valid_loss < best_loss
        if improved:
            best_loss, wait = valid_loss, 0
            save_model(model, classes, checkpoint)
        else:
            wait += 1
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "valid_loss": valid_loss, "valid_acc": valid_acc, "lr_head": lr,
            "seconds": time.perf_counter() - start, "best": improved,
        })
        logger.info("Epoca %2d  train loss %.3f acc %.3f  valid loss %.3f acc %.3f  lr %.0e  %.0f s%s",
                    epoch, train_loss, train_acc, valid_loss, valid_acc, lr, history[-1]["seconds"],
                    "  (mejor)" if improved else "")
        if wait >= config.early_stopping:
            logger.info("Early stopping en la epoca %d", epoch)
            break
    return history


def embed(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Salida de la capa previa al clasificador."""
    if hasattr(model, "embed"):
        return model.embed(x)
    # ResNet de torchvision, el mismo forward sin la capa fc.
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
    return torch.flatten(model.avgpool(x), 1)


@torch.no_grad()
def predict(model: nn.Module, dataset_path: Path, df: pd.DataFrame, image_size: int,
            batch_size: int = 128, num_workers: int = 0, embeddings: bool = False) -> np.ndarray:
    """Logits (o embeddings) de todas las imagenes de df, sin augmentation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = make_loader(BreedDataset(dataset_path, df, image_size), batch_size, False, num_workers, 0)
    outputs = []
    for x, _ in loader:
        x = x.to(device)
        outputs.append((embed(model, x) if embeddings else model(x)).cpu())
    model.cpu()
    return torch.cat(outputs).numpy()


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Matriz de confusion, filas = raza real y columnas = raza predicha."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_metrics(cm: np.ndarray) -> pd.DataFrame:
    """Precision, recall, specificity y F1 de cada clase, one-vs-rest."""
    total = cm.sum()
    rows = []
    for k in range(len(cm)):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        tn = total - tp - fp - fn
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        rows.append({"precision": precision, "recall": recall, "specificity": specificity(tn, fp), "f1": f1, "support": tp + fn})
    return pd.DataFrame(rows)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    """Accuracy y metricas macro (promedio por clase)."""
    per_class = per_class_metrics(confusion(y_true, y_pred, n_classes))
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(per_class["precision"].mean()),
        "recall": float(per_class["recall"].mean()),
        "specificity": float(per_class["specificity"].mean()),
        "f1": float(per_class["f1"].mean()),
    }
