"""Train baseline and/or transfer-learning RecycleVision classifiers."""
import argparse
import json

import pandas as pd
import tensorflow as tf

from src.config import MODELS_DIR, REPORTS_DIR, SEED
from src.data import class_weights, make_datasets
from src.evaluation import evaluate_predictions
from src.models import baseline_cnn, compile_model, transfer_model


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder containing one subfolder per waste class")
    parser.add_argument("--model", choices=["baseline", "mobilenetv2", "efficientnetb0"], default="mobilenetv2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    return parser.parse_args()


def main():
    args = arguments(); tf.keras.utils.set_random_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, classes = make_datasets(args.data_dir, args.batch_size)
    model = baseline_cnn(len(classes)) if args.model == "baseline" else transfer_model(len(classes), args.model)
    compile_model(model)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weights(args.data_dir, classes), callbacks=callbacks)
    if args.model != "baseline" and args.fine_tune_epochs:
        model.base_model.trainable = True
        for layer in model.base_model.layers[:-30]: layer.trainable = False
        compile_model(model, 1e-5)
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, class_weight=class_weights(args.data_dir, classes), callbacks=callbacks)
    y_true = tf.concat([labels for _, labels in val_ds], axis=0).numpy()
    probs = model.predict(val_ds, verbose=0)
    metrics = evaluate_predictions(y_true, probs, classes, REPORTS_DIR, args.model)
    model.save(MODELS_DIR / f"{args.model}.keras")
    (MODELS_DIR / f"{args.model}_labels.json").write_text(json.dumps(classes, indent=2))
    summary_path = REPORTS_DIR / "model_comparison.csv"
    frame = pd.DataFrame([metrics])
    pd.concat([pd.read_csv(summary_path), frame], ignore_index=True).drop_duplicates("model", keep="last").to_csv(summary_path, index=False) if summary_path.exists() else frame.to_csv(summary_path, index=False)
    print(f"Saved model and reports. Validation macro F1: {metrics['f1_macro']:.3f}")

if __name__ == "__main__": main()
