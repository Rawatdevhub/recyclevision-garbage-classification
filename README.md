# RecycleVision — Garbage Image Classification

An end-to-end computer-vision project for sorting waste images into recycling categories. It includes preprocessing and augmentation, a baseline CNN, MobileNetV2/EfficientNetB0 transfer learning, class-balanced training, evaluation reports, and a Streamlit interface.

## Dataset layout

Download one of the listed Kaggle datasets and arrange images into one folder per class. The code infers labels from the folder names, so it supports both the six-class TrashNet layout and expanded datasets.

```text
data/garbage/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

## Setup and training

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --data-dir data/garbage --model mobilenetv2 --epochs 15
streamlit run app.py
```

Available models are `baseline`, `mobilenetv2`, and `efficientnetb0`. Transfer-learning runs start with a frozen ImageNet backbone and then fine-tune its final layers at a lower learning rate. To compare alternatives, train each model; their validation metrics accumulate in `artifacts/reports/model_comparison.csv`. Select the model with the best macro F1 rather than accuracy alone.

## Outputs

- `artifacts/models/<model>.keras`: trained Keras model
- `artifacts/models/<model>_labels.json`: exact label mapping used at inference
- `artifacts/reports/*_metrics.csv`: accuracy, macro precision/recall/F1
- `artifacts/reports/*_classification_report.csv`: per-class scores
- `artifacts/reports/*_confusion_matrix.png`: visual error analysis

## Architecture

`Image upload → RGB resize (224×224) → augmentation during training → CNN/transfer-learning backbone → softmax class probabilities → top-three results`

Class weights automatically compensate for uneven image counts. The validation split is deterministic (seed 42), which makes model comparison fair. For deployment, train locally, commit neither the raw dataset nor large model artifacts, then deploy `app.py` on Streamlit Community Cloud with the selected model available to the app.

## Deployment

This repository includes Streamlit Cloud configuration (`runtime.txt` and `.streamlit/config.toml`) plus a GitHub Actions syntax check. Read [DEPLOYMENT.md](DEPLOYMENT.md) for the exact GitHub and Streamlit Cloud steps. A trained `.keras` model and its matching `_labels.json` file must be included through Git LFS or a model store before the cloud app can classify uploaded images.
