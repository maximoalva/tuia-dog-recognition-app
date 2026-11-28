from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SPLITS: tuple[str, ...] = ("train", "valid", "test")


def normalize_breed(name: str) -> str:
    """
    Normaliza el nombre de una raza: colapsa espacios repetidos y quita los de los extremos.
    
    Args:
        name: nombre de la raza (nombre de la carpeta).
        
    Returns:
        Nombre normalizado.
    """
    return " ".join(name.split())


def normalize_breed_folders(dataset_dir: Path, splits: tuple[str, ...] = DEFAULT_SPLITS) -> list[tuple[str, str, str]]:
    """
    Renombra las carpetas de raza cuyo nombre no está normalizado.

    Si la carpeta normalizada ya existe en el mismo split, mueve las imágenes dentro de ella y
    elimina la carpeta vieja. Falla si dos archivos tienen el mismo nombre, para no pisar datos.

    Args:
        dataset_dir (Path): carpeta raíz del dataset (contiene una subcarpeta por split).
        splits (tuple[str, ...]): splits a revisar. Los que no existen se ignoran.

    Returns:
        list[tuple[str, str, str]]: split, nombre_original, nombre_normalizado de las carpetas corregidas.
        Vacía si no hubo nada que corregir.
    """
    renamed: list[tuple[str, str, str]] = []
    for split in splits:
        split_dir = Path(dataset_dir) / split
        if not split_dir.is_dir():
            continue
        for breed_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            target_name = normalize_breed(breed_dir.name)
            if target_name == breed_dir.name:
                continue
            target = split_dir / target_name
            if target.exists():
                for f in breed_dir.iterdir():
                    if (target / f.name).exists():
                        raise FileExistsError(f"No se puede mover {f}: ya existe en {target}")
                    shutil.move(str(f), str(target / f.name))
                breed_dir.rmdir()
            else:
                breed_dir.rename(target)
            logger.info("%s: %r -> %r", split, breed_dir.name, target_name)
            renamed.append((split, breed_dir.name, target_name))
    return renamed
