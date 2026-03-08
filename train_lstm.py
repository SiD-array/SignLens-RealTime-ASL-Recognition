"""
ASL Action Recognition LSTM Training
===================================
Trains a TensorFlow/Keras LSTM model on 30-frame landmark sequences
saved as .npy clips, with labels defined in labels.json.

Usage:
    python train_lstm.py
"""

import sys
import io
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SEQUENCE_DIR = DATA_DIR / "sequences_lstm"
LABELS_PATH = PROJECT_ROOT / "labels.json"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "lstm_model.keras"

SEQUENCE_LENGTH = 30  # frames
FEATURE_SIZE = 63     # 21 landmarks × 3 coords

TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 25


# ============================================================================
# DATA LOADING
# ============================================================================

def load_labels() -> List[str]:
    """Load gesture labels from labels.json."""
    if not LABELS_PATH.exists():
        print(f"\n❌ Error: labels.json not found at {LABELS_PATH}")
        print("   Create labels.json as a JSON list of label strings, e.g.:")
        print('   ["Hello", "THANKYOU", "Sorry", "Yes", "No"]')
        sys.exit(1)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        print("\n❌ Error: labels.json must be a JSON list of label strings.")
        sys.exit(1)

    return labels


def load_sequence_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load 30-frame landmark sequences from SEQUENCE_DIR.

    Directory layout is expected to be:
        data/sequences_lstm/<LABEL>/*.npy

    Returns:
        X: array of shape (num_samples, 30, 63)
        y: array of shape (num_samples,) with integer label indices
        labels: list of label strings (index-aligned with y)
    """
    labels = load_labels()

    if not SEQUENCE_DIR.exists():
        print(f"\n❌ Error: sequence directory not found at {SEQUENCE_DIR}")
        print("   Run extract_landmarks.py first to generate .npy sequences.")
        sys.exit(1)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    print("\n" + "=" * 60)
    print("📂 LOADING SEQUENCE DATA")
    print("=" * 60)

    for label_idx, label in enumerate(labels):
        gesture_dir = SEQUENCE_DIR / label
        if not gesture_dir.exists():
            print(f"⚠️  No sequences found for label '{label}' at {gesture_dir}")
            continue

        npy_files = sorted(gesture_dir.glob("*.npy"))
        if not npy_files:
            print(f"⚠️  No .npy files for label '{label}'")
            continue

        for npy_path in npy_files:
            seq = np.load(npy_path)
            if seq.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                print(f"   ⚠️ Skipping {npy_path.name} with unexpected shape {seq.shape}")
                continue

            X_list.append(seq.astype(np.float32))
            y_list.append(label_idx)

        print(f"   ✅ Loaded {len(npy_files)} sequences for '{label}'")

    if not X_list:
        print("\n❌ Error: No valid sequences found.")
        sys.exit(1)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    print(f"\n📊 Total sequences: {len(X)}")
    print(f"📊 Sequence shape: {X.shape[1:]} (frames, features)")

    return X, y, labels


# ============================================================================
# MODEL DEFINITION
# ============================================================================

def build_lstm_model(num_classes: int) -> tf.keras.Model:
    """
    Build a simple LSTM-based action recognition model.

    Input shape is fixed to (SEQUENCE_LENGTH, FEATURE_SIZE) = (30, 63).
    """
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE), name="landmark_sequence")

    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="gesture_logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="asl_lstm_action_recognizer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n" + "=" * 60)
    print("🧠 LSTM MODEL SUMMARY")
    print("=" * 60)
    model.summary()

    return model


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("   🤟 ASL ACTION RECOGNITION - LSTM TRAINING")
    print("=" * 60)

    # Load data
    X, y, labels = load_sequence_data()

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n" + "=" * 60)
    print("📊 DATA SPLIT")
    print("=" * 60)
    print(f"   Training samples:   {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")

    # Build model
    model = build_lstm_model(num_classes=len(labels))

    # Train
    print("\n" + "=" * 60)
    print("🏋️  TRAINING LSTM MODEL")
    print("=" * 60)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("📈 FINAL EVALUATION")
    print("=" * 60)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"   Validation loss: {val_loss:.4f}")
    print(f"   Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

    # Save model
    print("\n" + "=" * 60)
    print("💾 SAVING MODEL")
    print("=" * 60)
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUTPUT_PATH)
    print(f"   Saved LSTM model to: {MODEL_OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nClasses ({len(labels)}): {labels}")
    print(f"Model ready for real-time use in main.py")


if __name__ == "__main__":
    main()

