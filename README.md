# DyrtyVision

YOLOv8-based waste classification system for sorting recyclables by material type.

## Class Taxonomy (9 classes)

| Index | Class            | Description             |
|-------|------------------|-------------------------|
| 0     | `cardboard`      | Cardboard / cartons     |
| 1     | `food_organics`  | Food / organic waste    |
| 2     | `glass`          | Glass bottles, jars     |
| 3     | `metal`          | Cans, foil, scrap metal |
| 4     | `misc_trash`     | Miscellaneous trash     |
| 5     | `paper`          | Paper, tissues, bags    |
| 6     | `plastic`        | Plastic bottles, bags   |
| 7     | `textile_trash`  | Textiles, shoes         |
| 8     | `vegetation`     | Plant / yard waste      |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
configs/        # YAML configuration files
src/
  data/         # Dataset download, conversion, splitting, validation
  augmentation/ # Albumentations pipeline and preview
  model/        # YOLOv8 config generation and export
  training/     # Training loop, callbacks, hyperparameter search
  evaluation/   # Metrics, visualization, benchmarking
  inference/    # FastAPI server, pre/post-processing, batch inference
  utils/        # Shared helpers (I/O, logging, config loader)
deploy/         # Dockerfiles, docker-compose, AWS infrastructure
tests/          # Unit and integration tests
scripts/        # One-off utility scripts
notebooks/      # Exploration notebooks
```

## Datasets

The pipeline downloads and merges annotations from:

- **RealWaste** — 4,752 images, 9 classes (1:1 mapping)
- **TrashNet / Kaggle Garbage Classification** — 2,527 images, remapped to 9 classes
- **TACO** — Trash Annotations in Context, 60 categories remapped to 9 classes

All labels are remapped to the 9-class taxonomy above and converted to YOLO format.

## Training

```bash
python -m src.training.train --config configs/train.yaml
```

## Inference

```bash
uvicorn src.inference.server:app --host 0.0.0.0 --port 8000
```

## Testing

```bash
pytest
```
