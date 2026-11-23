import albumentations as A
import cv2

def build_train_augmentation(image_size: int, seed: int | None = None) -> A.Compose:
    """
    Pipeline de augmentation para entrenamiento.
    Aplica transformaciones que simulan variaciones reales entre fotos de una misma
    raza sin alterar los rasgos que la distinguen.

    Args:
        image_size: Tamaño de salida, viene de IMAGE_SIZE definido en .env.
        seed: Fija la aleatoriedad (reproducibilidad).

    Returns:
        A.Compose: Pipeline de albumentations.
    """
    return A.Compose(
        [
            # Recorte aleatorio + leve estiramiento: robustez a encuadres parciales, escala y contextura del perro
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
            # Flip horizontal
            A.HorizontalFlip(p=0.5),
            # Rotaciones
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            # Blur de foco o movimiento
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 7))], p=0.2),
            # Variaciones de brillo y contraste
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            # Saturación y valor
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            # Ruido
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            # Compresión
            A.ImageCompression(quality_range=(45, 95), p=0.3),
            # Oclusiones parciales
            A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.10, 0.25), hole_width_range=(0.10, 0.25), fill=0, p=0.25),
            # Conversión a gris con baja probabilidad
            A.ToGray(p=0.05),
        ],
        seed=seed
    )
