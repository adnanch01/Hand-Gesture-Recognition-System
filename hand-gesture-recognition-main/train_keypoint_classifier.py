#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv, argparse, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

CSV = Path("model/keypoint_classifier/keypoint.csv")
KERAS_OUT = Path("model/keypoint_classifier/keypoint_classifier.keras")
TFLITE_OUT = Path("model/keypoint_classifier/keypoint_classifier.tflite")

def load(csv_path):
    X, y = [], []
    with open(csv_path, "r") as f:
        for row in csv.reader(f):
            *feat, lab = row
            X.append([float(x) for x in feat]); y.append(int(lab))
    return np.array(X, np.float32), np.array(y, np.int32)

def build(input_dim, n_classes):
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Normalization()(inp)
    x = tf.keras.layers.Dense(128, activation="relu")(x); x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x);  x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--keras_out", default=str(KERAS_OUT))
    ap.add_argument("--tflite_out", default=str(TFLITE_OUT))
    args = ap.parse_args()

    X, y = load(args.csv)
    n_classes = int(y.max()) + 1
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    m = build(X.shape[1], n_classes)
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs, batch_size=args.batch_size, verbose=2)

    m.save(args.keras_out)
    conv = tf.lite.TFLiteConverter.from_keras_model(m); tflite = conv.convert()
    Path(args.tflite_out).write_bytes(tflite)

    _, acc = m.evaluate(Xva, yva, verbose=0)
    print(f"Saved: {args.keras_out}, {args.tflite_out} | Val Acc: {acc:.4f}")

if __name__ == "__main__":
    main()
