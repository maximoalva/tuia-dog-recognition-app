# TP2 - Sistema de Detección y Clasificación de Razas de Perros

Plantilla base para desarrollar un pipeline completo de Computer Vision:

**Embeddings → Búsqueda por similitud → Clasificación → Detección → Pipeline completo**

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md) (versión PDF: [Trabajo Práctico N°2.pdf](<Trabajo Práctico N°2.pdf>)).

La cátedra provee la infraestructura completa (Docker, aplicación Gradio, base de datos vectorial, API, scripts y herramientas de evaluación y visualización). El grupo hace un **fork** de este repositorio, trabaja en una **feature branch** y completa **únicamente** las funciones indicadas en cada etapa. Cualquier modificación fuera de esas funciones debe justificarse en el Pull Request.

## Objetivo del backend

API asincrónica en Python que permite:

- Búsqueda por similitud (`/search`): imagen consultada, Top-K imágenes similares y raza predicha.
- Clasificación supervisada (`/classify`): raza y score con el modelo entrenado seleccionado.
- Detección + clasificación (`/detect`): bounding boxes, raza y scores de confianza.
- Consulta del estado del procesamiento asincrónico (`/status/{job_id}`).
- Selección dinámica del modelo de embeddings (`/models` + campo `model` en `/search`).

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-dog-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Etapa 2: entrenamiento, evaluación (incl. NDCG@10 de la Etapa 1) y comparación
├── ENUNCIADO.md                 # Enunciado completo del TP
├── Trabajo Práctico N°2.pdf     # Enunciado en PDF
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra (una pestaña por etapa)
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /search, /classify, /detect, /status, /models, /files
│       ├── bootstrap.py         # Construcción de los servicios (compartida por la API y los scripts)
│       ├── config.py            # Configuración vía .env
│       ├── files.py
│       ├── schemas.py
│       ├── evaluation/
│       │   └── metrics.py       # NDCG@k, precision/recall/F1, specificity
│       ├── visualization/
│       │   └── draw.py          # Dibujo de bounding boxes
│       ├── services/
│       │   ├── similarity_service.py   # ⚠️ Etapa 1: funciones a implementar
│       │   ├── classifier_service.py   # ⚠️ Etapa 2: funciones a implementar
│       │   ├── detection_service.py    # ⚠️ Etapa 3: funciones a implementar
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py      # Base vectorial en JSON
│           └── pgvector_store.py       # Base vectorial en PostgreSQL + pgvector
├── scripts/
│   ├── download_dataset.py      # Descarga el dataset de Kaggle
│   ├── build_index.py           # Indexa el dataset en la base vectorial
│   └── train_classifier.py      # Entrena/evalúa el clasificador (alternativa local a 2-train.ipynb)
├── data/
│   ├── dataset/                 # 70 Dog Breeds Image Dataset, no se versiona
│   └── embeddings.json          # Base vectorial JSON (si USE_PGVECTOR=false)
├── models/                      # Checkpoints entrenados (.pth / .onnx), no se versionan
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente

## Requisitos

- Python 3.12
- Docker (Docker Desktop incluye Docker Compose)
- Git
- Cuenta de Kaggle (para descargar el dataset)

## Dataset

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

```bash
python scripts/download_dataset.py
```

Requiere [credenciales de Kaggle](https://www.kaggle.com/docs/api) configuradas. Si no las tienen, descarguen el dataset manualmente desde Kaggle y descompriman `train/`, `valid/` y `test/` dentro de `data/dataset`.

## Configurá tus modelos

Entrená tus modelos (Etapa 2) con `2-train.ipynb` (en Google Colab, con GPU) y guardá los checkpoints dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `SIMILARITY_THRESHOLD`) para que apunte a tus modelos.

> El backend arranca aunque falten los checkpoints: `/health` informa cuáles encuentra en `models/`. Las pestañas de las Etapas 2 y 3 fallan hasta que existan.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

Es la forma en que la cátedra evalúa el trabajo.

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

Servicios disponibles: frontend en `http://localhost:8080`, backend en `http://localhost:8000` y PostgreSQL en `localhost:5432`.

### 2. Indexá la galería

Con la Etapa 1 implementada y el dataset en `data/dataset`, indexá el split de entrenamiento desde el contenedor del backend:

```bash
docker compose exec backend python /scripts/build_index.py --split train
```

## Opción 2 - Configurando el ambiente local

### 1. Cloná tu fork

```bash
git clone <url-de-tu-fork>
cd tuia-dog-recognition-app
git checkout -b feature/tp2-razas-perros
```

### 2. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/). También pueden usar `python3.12 -m venv .venv` y `pip`.

### 3. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 4. Activá el ambiente virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 5. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 6. Configurá las variables de entorno

Ajustá el archivo `.env.local.example` a tus necesidades y copialo a `src/.env`. El backend local y los scripts de `scripts/` leen la configuración desde ese archivo.

```bash
cp .env.local.example src/.env
```

### 7. Descargá el dataset

```bash
python scripts/download_dataset.py
```

### 8. Iniciá la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (por defecto). Con `USE_PGVECTOR=false` se usa la base vectorial JSON (`data/embeddings.json`) y no hace falta Docker para trabajar en local.

### 9. Iniciá el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 10. Iniciá el frontend (en otra terminal, con el ambiente virtual activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 11. Verificá que todo esté corriendo

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/docs` (Swagger) y `http://localhost:8000/health` (muestra el modelo seleccionado y los checkpoints encontrados)

## Scripts provistos

Se ejecutan desde la raíz del repositorio con el ambiente local activo (y PostgreSQL corriendo si `USE_PGVECTOR=true`). Leen la configuración de `src/.env` si existe:

```bash
# Descargar el dataset de Kaggle a data/dataset
python scripts/download_dataset.py

# Indexar la galería en la base vectorial (requiere la Etapa 1 implementada)
python scripts/build_index.py --split train
# ... o lo mismo desde el contenedor del backend (con docker compose up -d corriendo)
docker compose exec backend python /scripts/build_index.py --split train

# Entrenar y evaluar el clasificador localmente (alternativa a 2-train.ipynb)
python scripts/train_classifier.py --model resnet18_finetuned
```

Con `USE_PGVECTOR=false`, la indexación del split completo es lenta: el JSON se reescribe en cada inserción. Para pruebas rápidas, usen `--limit` (máximo de imágenes por raza).

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten el modelo de embeddings (`EMBEDDING_MODEL`, `EMBEDDING_DIM`), la búsqueda (`SIMILARITY_METRIC`, `SIMILARITY_THRESHOLD`, `TOP_K`), los checkpoints (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`), YOLO (`YOLO_MODEL`, `YOLO_CONF_THRESHOLD`, `YOLO_DOG_CLASS_ID`) y los paths.
3. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

| Endpoint | Descripción |
|----------|-------------|
| `POST /upload` | Sube una imagen al servidor |
| `POST /search` | Etapa 1: búsqueda por similitud (acepta `model` y `top_k`) |
| `POST /classify` | Etapa 2: clasificación con el modelo entrenado (acepta `model`) |
| `POST /detect` | Etapa 3: detección + clasificación |
| `GET /status/{job_id}` | Estado del procesamiento asincrónico |
| `GET /models` | Modelos de embeddings disponibles |
| `GET /health` | Estado del backend y checkpoints encontrados |

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/search`, `/classify`, `/detect`, `/status`, `/models`) y manejo de jobs.
- Orquestación de cada etapa: `search` e `index_image` (Etapa 1), `classify_image` (Etapa 2) y `predict` (Etapa 3: detección → recorte → clasificación → JSON).
- Selección dinámica del modelo de embeddings: `baseline`, `resnet18_finetuned` o `cnn_custom`.
- Carga de checkpoints (`.pth` / `.onnx`) desde `models/` y selección del modelo activo.
- Similitud configurable (`cosine` o `l2`) y manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Funciones auxiliares de evaluación (`lib/evaluation/metrics.py`) y de visualización (`lib/visualization/draw.py`).
- Frontend Gradio con una pestaña por etapa: Etapa 1 (imagen consultada, Top-K similares, raza predicha y selección de modelo), Etapa 2 (raza y score con el modelo entrenado) y Etapa 3 (bounding boxes, razas y scores).

### A implementar por el grupo

En el repositorio base, estas funciones lanzaban `NotImplementedError`:

| Etapa | Función | Archivo |
|-------|---------|---------|
| 1 | `extract_embedding(image)` | `src/lib/services/similarity_service.py` |
| 1 | `search_similar_images(embedding, top_k)` | `src/lib/services/similarity_service.py` |
| 1 | `predict_breed_from_neighbors(results)` | `src/lib/services/similarity_service.py` |
| 2 | `train_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `evaluate_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `extract_custom_embedding(image)` | `src/lib/services/classifier_service.py` |
| 3 | `detect_dogs(image)` | `src/lib/services/detection_service.py` |
| 3 | `classify_detected_dog(crop)` | `src/lib/services/detection_service.py` |

En las Etapas 1 y 3 el trabajo se limita a completar las funciones. En la Etapa 2, además, el grupo entrena sus propios modelos (Modelo A: fine-tuning de ResNet18/ResNet50; Modelo B: CNN propia, **ambos obligatorios**) y hace un estudio de resultados.

Ver el contrato de cada función, recomendaciones y errores frecuentes en [ENUNCIADO.md](ENUNCIADO.md#24-funciones-a-implementar).

### Implementación del grupo

Las tres etapas usan el mismo preprocesamiento (`src/lib/preprocessing.py`, ver [Preprocesamiento y data augmentation](#preprocesamiento-y-data-augmentation)), tanto en las notebooks como en la app.

**Etapa 1 — Búsqueda por similitud** (`similarity_service.py`)

- `extract_embedding`: ResNet18 preentrenada en ImageNet (`IMAGENET1K_V1`) sin la capa de clasificación (`fc = Identity`). Devuelve el vector de 512 del *global average pooling*, normalizado L2. La red se carga una sola vez.
- `search_similar_images`: con pgvector y `cosine`, la búsqueda la resuelve la base (`store.search`). Con `l2` o con el *store* JSON, se recorren todos los registros y se calcula el score con `similarity`. Devuelve los `top_k` vecinos ordenados por score descendente.
- `predict_breed_from_neighbors`: voto ponderado por score (cada vecino suma su score a su raza, y en caso de empate gana la raza del vecino más parecido). Devuelve la raza ganadora y, como score, la similitud media de sus vecinos. Si el mejor vecino queda por debajo de `SIMILARITY_THRESHOLD`, devuelve `unknown`.

**Etapa 2 — Clasificación** (`classifier_service.py`, con la lógica de entrenamiento en `src/lib/training.py` y la CNN propia en `src/lib/models/cnn_custom.py`)

- `train_classifier`: entrena el modelo activo con los splits de `data/processed/splits.csv` y el augmentation solo en `train`. Usa *early stopping* por la loss de `valid` y guarda el mejor checkpoint como modelo completo, con la lista de razas en `model.classes`, más el historial por época en `output/history_<checkpoint>.json`. Los hiperparámetros están en `TRAIN_CONFIGS` (`training.py`).
- `evaluate_classifier`: accuracy, y precision, recall, specificity y F1 macro (one-vs-rest por raza, promediadas) sobre `test`.
- `extract_custom_embedding`: embedding de la capa previa al clasificador del modelo activo, normalizado L2.

**Etapa 3 — Detección y clasificación** (`detection_service.py`)

- `detect_dogs`: YOLO11n preentrenado en COCO, filtrando por la clase `dog` (`YOLO_DOG_CLASS_ID`) y por `YOLO_CONF_THRESHOLD` dentro del detector. Devuelve las cajas en píxeles enteros, recortadas a los bordes de la imagen y ordenadas por confianza. Descarta las cajas que quedan casi enteras (≥ 85 % de su área) dentro de otra caja más confiable, porque en las pruebas aparecían cajas duplicadas sobre un mismo perro. El detector se carga una sola vez y la inferencia se serializa con un lock, porque el predictor de ultralytics guarda estado interno.
- `classify_detected_dog`: aplica al recorte el mismo preprocesamiento del entrenamiento, corre el modelo activo (`.pth` u `.onnx`) y devuelve la raza con mayor probabilidad (*softmax*) y esa probabilidad. La lista de razas sale del checkpoint (`model.classes` o la metadata del ONNX), o de `data/processed/classes.json` si el modelo no la trae.

### Orden de trabajo sugerido

1. `1-pre-processing.ipynb`: EDA, filtrado, conjunto independiente, preprocesamiento, augmentation y splits.
2. Etapa 1 en `similarity_service.py`. Indexá la galería con `python scripts/build_index.py --split train` y probá la pestaña de la Etapa 1.
3. Etapa 2 en `classifier_service.py`, entrenando y comparando los modelos con `2-train.ipynb`. Descargá los checkpoints a `models/`.
4. Etapa 3 en `detection_service.py` y validación end-to-end con `docker compose up`.

## Notebooks

| Notebook | Contenido |
|----------|-----------|
| `1-pre-processing.ipynb` | Dataset, EDA, control de calidad, conjunto independiente, preprocesamiento, data augmentation y splits |
| `2-train.ipynb` | Evaluación de la búsqueda (NDCG@10), entrenamiento del Modelo A y del Modelo B, métricas, análisis de errores, comparación y exportación de checkpoints |

Se pueden entregar commiteadas en el repositorio o en Google Colab, con el link a la notebook de Colab en una celda Markdown al inicio de la notebook del repositorio. En ambos casos deben estar **ejecutadas y con las salidas visibles**: una notebook sin salidas se considera **no entregada** (ver [ENUNCIADO.md](ENUNCIADO.md#82-notebooks-repositorio-o-google-colab)).

## Dataset

El análisis completo está en [`1-pre-processing.ipynb`](1-pre-processing.ipynb). Los splits finales quedan en `data/processed/splits.csv` (ruta, raza, split e índice de clase) y la lista de razas, en el orden de esos índices, en `data/processed/classes.json`.

### Splits

Se usan los **splits originales del dataset** (*70 Dog Breeds Image Dataset*, Kaggle), con el filtrado de calidad aplicado a `train`.

| Split | Imágenes | Razas | Imágenes por raza (mín–máx) | Media por raza |
|---|---|---|---|---|
| `train` | 7539 | 70 | 61–187 | 107,7 |
| `valid` | 686 | 70 | 8–10 | 9,8 |
| `test` | 700 | 70 | 10–10 | 10,0 |
| **Total** | **8925** | 70 | | |

Antes del filtrado, `train` tenía 7946 imágenes (entre 65 y 198 por raza), y `valid` y `test`, 700 cada uno. El desbalance de `train` es moderado y se mantiene después del filtrado. `test` no se modifica para que la evaluación sea comparable con el dataset original.

**Corrección de etiquetas:** la raza *American Spaniel* figura como `American  Spaniel` (con dos espacios) en `valid`, lo que generaba 71 etiquetas en lugar de 70. Se corrige renombrando las carpetas (`normalize_breed_folders`, en `src/lib/dataset.py`), se aplica tras descargar el dataset.

### Reglas de filtrado y calidad

Las reglas de calidad se aplican **solo a `train`**: `valid` y `test` se conservan completos para que la evaluación refleje imágenes reales.

| Regla | Criterio | Imágenes descartadas |
|---|---|---|
| Archivos corruptos | La imagen no se puede decodificar completa | 0 |
| Baja resolución | Ancho o alto menor a 64 px | 0 |
| Exposición extrema | Brillo medio menor a 10 o mayor a 245 | 0 |
| Imágenes borrosas | Varianza del Laplaciano menor a 25,6 | 3 |
| Duplicados y casi-duplicados | Hash perceptual (pHash) a distancia de Hamming ≤ 8 bits | 391 |
| Etiqueta contradictoria | La misma foto en dos razas distintas (revisión manual) | 15 |
| Exclusión manual | No es un perro real (una remera estampada) | 1 |
| **Total** | Una imagen puede incumplir más de una regla | **407** |

Además, se descartaron de `valid` **14 imágenes** que tenían una copia en `test`, para que no haya fuga de datos entre `valid` y `test`.

Hallazgos principales:
- **Fuga de datos entre splits:** el **9,4 %** de `valid` y el **9,7 %** de `test` tenían una copia en `train`. Se descartan las copias de `train`; después del filtrado no queda ninguna imagen repetida entre splits.
- **Ruido de etiquetas:** 12 pares de copias aparecen con dos razas distintas, casi todos entre razas muy parecidas (*American Spaniel* / *Irish Spaniel*, *Lhasa* / *Shih-Tzu*, *Cockapoo* / *Labradoodle*).
- **Umbrales elegidos mirando ejemplos:** el brillo y la nitidez dependen del color y del tipo de pelo de cada raza, así que umbrales más estrictos habrían eliminado perros bien fotografiados de razas oscuras, blancas o de pelo esponjoso.

### Conjunto independiente de evaluación

- **Fuente:** fotos propias, aportadas por los dueños de los perros, más fotos de **Wikimedia Commons** para completar las razas que no conseguimos con fotos propias. Solo perros de raza, para que la etiqueta sea confiable. Ninguna foto muestra personas identificables.
- **Licencia:** los dueños dieron su consentimiento para usar y mostrar sus fotos en este trabajo académico. Las de Wikimedia Commons son de **dominio público**. Las imágenes no se suben al repositorio.
- **Link de descarga (solo lectura):** [conjunto independiente en Google Drive](https://drive.google.com/file/d/186EM7sIrJPj4XkEUVLZI6UH_I3amImYO/view?usp=sharing). Se descomprime en `data/dataset/independent/`.
- **Estructura:** una carpeta por raza, con el mismo nombre que en el dataset, más `_Escenas/` (varios perros, o un perro con otros animales u objetos) y `_Sin perros/` (por ejemplo, gatos).

| Tipo | Imágenes |
|---|---|
| Un perro | 124 |
| Escenas | 6 |
| Sin perros | 6 |
| **Total** | **136** |

Las fotos de un perro cubren **20 razas**, entre 5 y 14 por raza: *Border Collie* (14), *Greyhound* (13), *Boxer* (8), *German Sheperd* (7), *Labrador* y *Maltese* (6), y *Beagle*, *Bearded Collie*, *Chihuahua*, *Dalmation*, *Doberman*, *French Bulldog*, *Golden Retriever*, *Japanese Spaniel*, *Pit Bull*, *Poodle*, *Rottweiler*, *Schnauzer*, *Shih-Tzu* y *Siberian Husky* (5 cada una). Incluyen razas fáciles de confundir (*Labrador* / *Golden Retriever*, *Shih-Tzu* / *Maltese*, *Border Collie* / *Bearded Collie*).

El 70 % de las fotos son verticales (la mayoría sacadas con el celular), solo el 5 % son cuadradas y la resolución va de 271 a 5184 px de ancho, a diferencia del dataset (todas de 224×224): es el cambio de dominio que este conjunto permite medir. Comparadas con el hash perceptual contra todo el dataset, ninguna está repetida (la imagen más parecida queda a 12 bits o más, por encima del umbral de 8).

### Preprocesamiento y data augmentation

El preprocesamiento está en `src/lib/preprocessing.py` y lo usan tanto las notebooks como los servicios de la app (`preprocess(image, size, augment=False)`), para que el modelo reciba lo mismo al entrenar y en producción:

1. **Orden de canales:** la app entrega imágenes **BGR** (OpenCV), que se convierten a **RGB**. El entrenamiento también lee las imágenes con OpenCV, así que hay un solo camino.
2. **Resize con padding** a `IMAGE_SIZE` × `IMAGE_SIZE` (224): se respeta la proporción y se rellena con el color medio de ImageNet, que después de normalizar vale prácticamente 0. Se eligió frente al resize directo, que deforma al perro en fotos verticales o recortes alargados de YOLO.
3. **Normalización con las estadísticas de ImageNet** para todos los modelos, incluida la CNN propia: las de `train` son muy parecidas (diferencias menores a 0,05).

El **data augmentation** está en `src/lib/augmentation.py` y se aplica **solo a `train`**, antes del resize y con seed fija:

| Transformación | Parámetros | Prob. |
|---|---|---|
| Recorte aleatorio | 70–100 % del área, relación de aspecto 0,9–1,1 | 1 |
| Flip horizontal | — | 0,5 |
| Rotación | ±15°, bordes en espejo | 0,6 |
| Blur gaussiano o de movimiento | kernel de 3–5 / 3–7 px | 0,2 |
| Brillo y contraste | ±25 % | 0,6 |
| Saturación y valor (sin cambio de tono) | ±15 / ±10 | 0,3 |
| Ruido gaussiano | desvío del 1–5 % | 0,2 |
| Compresión JPEG | calidad 45–95 | 0,3 |
| Oclusiones | 1–2 rectángulos de 10–25 % del lado | 0,25 |
| Escala de grises | — | 0,05 |

No se aplican flips verticales, rotaciones grandes, cambios de tono ni deformaciones fuertes, porque no ocurren en fotos reales o alteran rasgos que distinguen a las razas (el color del pelaje y las proporciones del cuerpo).

## Modelos y entrenamiento

El entrenamiento, los experimentos y la elección de cada modelo están en [`2-train.ipynb`](2-train.ipynb).

### Baseline de embeddings (Etapa 1)

**ResNet18** preentrenada en ImageNet, sin la capa de clasificación: embedding de **512**. Elegimos ResNet18 porque es la misma red del Modelo A, así la comparación entre el baseline y el Modelo A mide solo el efecto del entrenamiento, y porque es más liviana que ResNet50. ImageNet incluye unas 120 razas de perro, así que sus embeddings ya separan bien las razas sin entrenar.

### Modelo A — ResNet18 preentrenada

Se reemplaza la última capa por una lineal con una salida por raza (70), y el embedding sigue siendo de 512. Comparamos dos estrategias:

| | Feature extraction (congelada) | Fine-tuning completo |
|---|---|---|
| Qué se entrena | Solo la última capa (35.910 parámetros); el resto de la red y BatchNorm quedan congelados | Toda la red (11,2 M de parámetros) |
| Learning rate | 1e-3 | 1e-3 en la última capa y 1e-4 en el resto |
| Checkpoint | `resnet18_frozen.pth` | `resnet18_finetuned.pth` |

Hiperparámetros comunes: AdamW con *weight decay* de 1e-4, *label smoothing* de 0,1, batch de 64 y hasta 30 épocas. El *learning rate* se divide por 10 si la loss de `valid` no mejora en 2 épocas (`ReduceLROnPlateau`), y el entrenamiento se corta si no mejora en 5 (*early stopping*), conservando el checkpoint de la mejor época. Usa el augmentation de la Parte 1, solo en `train`.

**La app usa la red congelada.** En `valid` gana el fine-tuning (96,6 % contra 94,2 %), pero en `test` empatan (95,4 %), y con las fotos reales del conjunto independiente la congelada acierta el 62,1 % contra 44,4 %. Al reentrenar toda la red con los recortes del dataset, el fine-tuning se ajustó a ese encuadre y pierde con fotos con fondo. La app sigue llamando a este modelo `resnet18_finetuned`, que es el nombre que le da el código, pero `RESNET18_MODEL_NAME` apunta a `resnet18_frozen.pth`.

### Modelo B — CNN propia

Definida en `src/lib/models/cnn_custom.py` (`CNNCustom`), para que `torch.load` pueda importarla desde la app:

- Bloques de convoluciones 3×3 sin bias, cada una con BatchNorm y ReLU, y un MaxPool que reduce la imagen a la mitad.
- *Global average pooling* en lugar de aplanar (con 256 mapas de 14×14, aplanar daría 25 M de parámetros en la capa siguiente).
- Capa de embedding de 512 (`EMBEDDING_DIM`) con ReLU, para usarla en la búsqueda, *dropout* de 0,3 y un clasificador lineal de 70 salidas.

Se entrena desde cero con *learning rate* 1e-3 en toda la red y los mismos optimizador, *label smoothing* y augmentation que el Modelo A.

| Experimento | Cambio | Parámetros | Valid acc | Valid loss |
|---|---|---|---|---|
| E1 | 4 bloques de 1 convolución (32–256 canales), 50 épocas | 556 mil | 43,6 % | 2,438 |
| E2 | 5 bloques de 2 convoluciones (32–512 canales), 50 épocas | 5,0 M | 78,9 % | 1,485 |
| E3 | E2 sin augmentation | 5,0 M | 69,2 % | 1,738 |
| **E4** | **E2 con hasta 100 épocas y más paciencia (5 para el LR y 10 para el early stopping)** | **5,0 M** | **88,2 %** | **1,198** |

El Modelo B es **E4**. Lo que más sumó fue la capacidad (E1 → E2), después entrenar más tiempo (E2 → E4), y sin augmentation la red sobreajusta (E3).

### Checkpoints

Los checkpoints no se versionan. Están en [esta carpeta de Google Drive](https://drive.google.com/drive/folders/1ZBhlvHyVXPPI1jijxQv397fIiwk9_iou) (solo lectura) y se descargan en `models/`:

| Archivo | Uso |
|---|---|
| `resnet18_frozen.pth` | Modelo A de la app (`RESNET18_MODEL_NAME`) |
| `cnn_custom.pth` | Modelo B (`CNN_CUSTOM_MODEL_NAME`) |
| `resnet18_finetuned.pth` | Fine-tuning completo, evaluado en la notebook 2 (la app no lo usa) |

Se guardan como modelo completo (`torch.save(model)`), con la lista de razas en `model.classes`. Los pesos del baseline (torchvision) y de YOLO se descargan solos en el primer uso.

### Detector (Etapa 3)

**YOLO11n** preentrenado en COCO (clase `dog` = 16), sin entrenar. Lo elegimos frente a YOLOv8n en `1-pre-processing.ipynb` (sección 4.7): sobre las imágenes del dataset, las que quedan sin perro detectado bajan de 44,7 % a 31,6 %, con una latencia en CPU del mismo orden y menos parámetros (2,6 M contra 3,2 M).

`YOLO_CONF_THRESHOLD`: **0,25**. **[Pendiente: justificar con la comparación de 2 o 3 umbrales.]**

### Umbral de similitud (Etapa 1)

`SIMILARITY_THRESHOLD=0.848` (`cosine`), calibrado en `valid`. Como en `valid` todas las razas están en la galería, simulamos razas desconocidas buscando cada consulta en la galería sin las imágenes de su raza. Con eso se calcula, para cada umbral, el FRR (perros de razas conocidas rechazados) y el FAR (razas desconocidas aceptadas). Comparamos el EER (0,876, con un 21 % de error de cada lado) con un FRR fijo del 5 % (0,848, con un FAR de 52,5 %). Elegimos el **FRR 5 %**, porque rechazar un perro de una de las 70 razas, el caso de uso principal, es el error más molesto. Con el EER, el 23 % de las consultas de `test` y el 87 % de las fotos reales saldrían como `unknown`.

### Configuración de la app

Valores de `.env.local.example` y `.env.docker.example`:

| Variable | Valor |
|---|---|
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `baseline` / `512` |
| `SIMILARITY_METRIC` / `SIMILARITY_THRESHOLD` | `cosine` / `0.848` |
| `TOP_K` | `10` (el mejor en `valid`, 93,6 % de accuracy) |
| `RESNET18_MODEL_NAME` | `resnet18_frozen.pth` |
| `CNN_CUSTOM_MODEL_NAME` | `cnn_custom.pth` |
| `YOLO_MODEL` / `YOLO_CONF_THRESHOLD` | `yolo11n.pt` / `0.25` |

La base vectorial entregada queda indexada con los embeddings del **baseline** (`python scripts/build_index.py --split train`). La red congelada da exactamente los mismos embeddings que el baseline (comparten todas las capas hasta el embedding), así que la búsqueda y la clasificación usan la misma representación.

## Resultados

Todas las métricas salen de [`2-train.ipynb`](2-train.ipynb). `test` tiene 700 imágenes (10 por raza) y el conjunto independiente, 124 fotos de un perro.

### Etapa 1 — Búsqueda por similitud

Galería: `train` (7539 imágenes). Consultas: `test` y el conjunto independiente. `TOP_K=10`, sin umbral.

| Embeddings | Dim | NDCG@10 test | NDCG@10 indep. | Precision@10 test | Precision@10 indep. | Accuracy test | Accuracy indep. |
|---|---|---|---|---|---|---|---|
| **Baseline (ResNet18 ImageNet)** | 512 | 0,873 | **0,572** | 0,860 | **0,568** | 0,934 | **0,621** |
| Modelo A, fine-tuning | 512 | **0,953** | 0,476 | **0,952** | 0,479 | **0,966** | 0,516 |
| Modelo B, CNN propia | 512 | 0,836 | 0,222 | 0,829 | 0,223 | 0,861 | 0,234 |

La red congelada del Modelo A da los mismos embeddings que el baseline. `cosine` y `l2` dan exactamente los mismos resultados, porque con embeddings de norma 1 ordenan los vecinos igual. Solo cambia la escala del score. Elegimos los embeddings del **baseline**: en `test` ganan los del fine-tuning, pero la pestaña de búsqueda recibe la foto entera del usuario, y con fotos reales el baseline anda mejor.

### Etapa 2 — Clasificación

| Métrica | Modelo A congelado (app) | Modelo A fine-tuning | Modelo B (CNN propia) |
|---|---|---|---|
| Accuracy valid | 0,942 | **0,966** | 0,882 |
| Accuracy test | **0,954** | **0,954** | 0,880 |
| Precision / Recall / F1 test (macro) | 0,957 / 0,954 / 0,953 | 0,957 / 0,954 / 0,954 | 0,891 / 0,880 / 0,879 |
| Specificity test (macro) | 0,999 | 0,999 | 0,998 |
| Accuracy independiente (foto entera) | **0,621** | 0,444 | 0,250 |
| F1 independiente (macro, 20 razas) | **0,691** | 0,523 | 0,325 |
| Accuracy independiente (recorte de YOLO) | **0,702** | 0,669 | 0,452 |
| Parámetros / tamaño | 11,2 M / 44,9 MB | 11,2 M / 44,9 MB | 5,0 M / 20,1 MB |
| Tiempo por imagen en CPU | 21 ms | 21 ms | 29 ms |
| Tiempo de entrenamiento (GPU) | 5 min | 10 min | 100 min |

- **Modelo A vs. Modelo B:** el A gana en todas las métricas de desempeño. La ResNet trae de ImageNet filtros para bordes, texturas y partes de animales, mientras que la CNN propia tiene que aprenderlos con unas 108 fotos por raza. El B pesa la mitad, pero es más lento: no reduce la resolución tan rápido como la ResNet en las primeras capas y hace casi el doble de operaciones.
- **Errores en `test`:** casi todos entre razas parecidas. El par más confundido del A (fine-tuning) es *Boston Terrier* → *Bulldog* (5 de 32 errores), y el del B, *Cockapoo* → *Labradoodle* (5 de 84). El B además confunde razas que no se parecen tanto (*Saint Bernard* con *Blenheim*, *Basset* o *Boxer*), probablemente porque se apoya más en el color que en la forma.
- **Cambio de dominio:** todos los modelos caen mucho con fotos reales. Recortar el perro con YOLO recupera buena parte, así que el fondo explica una parte importante de la caída. Lo que queda viene de la pose, la luz y el color: en las fotos reales hay errores con score alto que en `test` casi no aparecen.

### Búsqueda por similitud vs. clasificación supervisada

La **búsqueda** no necesita entrenamiento: para sumar una raza alcanza con indexar fotos de esa raza. Además muestra imágenes parecidas que explican la predicción, y con el umbral puede responder `unknown`. Como contrapartida, depende de la galería y sus embeddings no están ajustados a las razas, así que en `test` acierta menos que el clasificador (93,4 % contra 95,4 %). La **clasificación** es más precisa con imágenes parecidas a las de entrenamiento y no necesita una base de datos, pero las clases quedan fijas y siempre elige alguna de las 70, incluso con score alto cuando se equivoca. Conviene clasificar cuando las clases son fijas y las imágenes se parecen a las de entrenamiento (como los recortes de la Etapa 3), y buscar cuando las clases cambian, cuando sirve mostrar ejemplos o cuando hay que poder decir que la imagen no se parece a nada conocido.

## Problemas encontrados y soluciones

| Problema | Solución |
|---|---|
| La raza *American Spaniel* figuraba con dos espacios en `valid`: 71 etiquetas en lugar de 70, y una clase de `valid` que no existía en `train`. | `normalize_breed_folders` (`src/lib/dataset.py`) renombra las carpetas en disco. Se llama en `scripts/download_dataset.py` y en la notebook 1. |
| **Fuga de datos:** el 9,4 % de `valid` y el 9,7 % de `test` tenían una copia en `train`, y había 14 copias entre `valid` y `test`. | Detección de casi-duplicados con hash perceptual. Se descartan las copias de `train`, y de cada par entre `valid` y `test` se descarta la de `valid`. Una verificación confirma que no queda ninguna copia entre splits. |
| 12 fotos aparecían repetidas con dos razas distintas. | Revisión manual de cada par: se conserva la copia con la raza correcta, o se descartan las dos si no se puede saber. |
| YOLO, entrenado con escenas completas, no detectaba perros en los recortes cerrados del dataset: los confundía con osos, ovejas o gatos. | Cambiamos YOLOv8n por YOLO11n (las imágenes sin perro detectado bajan de 44,7 % a 31,6 %), y "YOLO no detecta un perro" no se usa como regla de descarte. |
| YOLO a veces devolvía dos cajas sobre el mismo perro. | `detect_dogs` descarta las cajas que quedan casi enteras dentro de otra más confiable. |
| No había ejemplos de razas desconocidas para calibrar el umbral de `unknown`. | Se simulan en `valid`, buscando cada consulta en la galería sin las imágenes de su raza (ver [Umbral de similitud](#umbral-de-similitud-etapa-1)). |
| La CNN propia aprendía muy despacio y llegaba al tope de épocas todavía mejorando. | Más capacidad (5 bloques de 2 convoluciones), más épocas y más paciencia para el *scheduler* y el *early stopping* (E2 → E4: de 78,9 % a 88,2 % en `valid`). |
| El fine-tuning ganaba en `valid` pero perdía mucho con fotos reales: `valid` y `test` salen del mismo dataset de recortes y no muestran ese problema. | Comparamos las dos estrategias también con el conjunto independiente y la app usa la red congelada (ver [Modelo A](#modelo-a--resnet18-preentrenada)). Lo ideal sería un conjunto de validación con fotos reales. |
| El checkpoint necesita la lista de razas para traducir el índice predicho. | Se guarda en `model.classes` antes de `torch.save`, y `data/processed/classes.json` (versionado) queda como respaldo. |

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- `EMBEDDING_DIM` debe coincidir con la dimensión del embedding del modelo elegido. Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- La base vectorial guarda los embeddings de un modelo a la vez: al cambiar de modelo de embeddings hay que reindexar (`scripts/build_index.py`).
- Las rutas de la galería se guardan relativas a `DATA_PATH` (por ejemplo, `dataset/train/<raza>/<imagen>.jpg`): una base indexada desde el entorno local también sirve para el backend en Docker, porque comparten PostgreSQL y la carpeta `data/`.
- Los modelos (`*.pth`, `*.pt`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen los checkpoints en `models/` y súbanlos a un link **público de solo lectura** para que los docentes puedan descargarlos.
- Los pesos preentrenados (torchvision/timm y YOLO) se descargan en el primer uso, por lo que se necesita conexión a internet.

## Entrega

- Pull Request desde su feature branch hacia `main` **de su propio repositorio**, **sin mergear**, con el link subido al campus. No abran el PR contra el repositorio de la cátedra.
- Si el repositorio es privado, los docentes deben tener acceso.
- Notebooks `1-pre-processing.ipynb` y `2-train.ipynb` ejecutadas, con sus salidas.
- Para aprobar, el sistema debe andar sin errores con:

```bash
docker compose build
docker compose up
```

**Fecha de entrega:** a definir por la cátedra. Solo se consideran los commits hasta esa fecha y hora; se recomienda trabajar de forma incremental.

El detalle del contenido del PR, la plantilla de descripción y el checklist están en [ENUNCIADO.md](ENUNCIADO.md#8-entrega-pull-request).
