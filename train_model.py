"""
ASL Gesture Classification Model Training
==========================================
Trains a RandomForestClassifier on extracted hand landmarks.

Usage:
    python train_model.py
"""

import sys
import io
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed_landmarks"
CSV_PATH = DATA_DIR / "gesture_landmarks.csv"
MODEL_PATH = PROJECT_ROOT / "sign_language_model.pkl"

# Training parameters
TEST_SIZE = 0.20  # 20% for testing
RANDOM_STATE = 42  # For reproducibility


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(csv_path: Path):
    """
    Load CSV and split into features (X) and labels (y).
    
    Args:
        csv_path: Path to the gesture landmarks CSV
    
    Returns:
        X: Feature matrix (landmark coordinates)
        y: Label vector (gesture names)
        feature_names: List of feature column names
    """
    print(f"📂 Loading data from: {csv_path.name}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"   Total samples: {len(df)}")
    print(f"   Total features: {len(df.columns) - 1}")
    
    # Split into features and labels
    # All columns except the last one are features
    X = df.iloc[:, :-1].values  # All columns except 'label'
    y = df.iloc[:, -1].values   # Last column is 'label'
    
    feature_names = df.columns[:-1].tolist()
    
    # Print class distribution
    print("\n📊 Class distribution:")
    class_counts = pd.Series(y).value_counts()
    for label, count in class_counts.items():
        print(f"   {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    return X, y, feature_names


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained classifier
    """
    print("\n🌲 Training Random Forest Classifier...")
    
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=100,       # Number of trees
        max_depth=None,         # No max depth limit
        min_samples_split=2,    # Minimum samples to split
        min_samples_leaf=1,     # Minimum samples per leaf
        random_state=RANDOM_STATE,
        n_jobs=-1,              # Use all CPU cores
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    print("✅ Model trained successfully!")
    print(f"   Number of trees: {model.n_estimators}")
    print(f"   Number of features: {model.n_features_in_}")
    print(f"   Number of classes: {len(model.classes_)}")
    print(f"   Classes: {list(model.classes_)}")
    
    return model


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print metrics.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
    """
    print("\n" + "=" * 60)
    print("📈 MODEL EVALUATION")
    print("=" * 60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n📋 Classification Report:")
    print("-" * 60)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    
    # Confusion matrix
    print("🔢 Confusion Matrix:")
    print("-" * 60)
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Print header
    print(f"{'Predicted →':>12}", end="")
    for label in labels:
        print(f"{label:>10}", end="")
    print()
    print("-" * (12 + 10 * len(labels)))
    
    # Print rows
    for i, label in enumerate(labels):
        print(f"{'Actual ↓':>12}" if i == 0 else f"{'':>12}", end="")
        print(f"{label:>10}", end=" | ")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>8}", end="")
        print()
    
    print("-" * 60)
    
    # Identify struggling classes
    print("\n⚠️  Performance Analysis:")
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    struggling = []
    for label in labels:
        if label in report_dict:
            f1 = report_dict[label]['f1-score']
            if f1 < 0.8:
                struggling.append((label, f1))
    
    if struggling:
        print("   Classes that may need more training data:")
        for label, f1 in sorted(struggling, key=lambda x: x[1]):
            print(f"   - {label}: F1-score = {f1:.3f}")
    else:
        print("   ✅ All classes performing well (F1 > 0.80)")
    
    return accuracy


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model(model, model_path: Path):
    """
    Save the trained model using pickle.
    
    Args:
        model: Trained classifier
        model_path: Path to save the model
    """
    print(f"\n💾 Saving model to: {model_path.name}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Verify the save
    file_size = model_path.stat().st_size / 1024  # KB
    print(f"✅ Model saved successfully! ({file_size:.1f} KB)")
    
    # Test loading the model
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    print(f"✅ Model verified - can be loaded successfully")
    print(f"\n📝 To use this model in your live script:")
    print(f"   import pickle")
    print(f"   with open('{model_path.name}', 'rb') as f:")
    print(f"       model = pickle.load(f)")
    print(f"   prediction = model.predict(landmarks)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("   🤟 ASL GESTURE MODEL TRAINING")
    print("=" * 60)
    
    # Check if CSV exists
    if not CSV_PATH.exists():
        print(f"\n❌ Error: CSV not found at {CSV_PATH}")
        print("   Run extract_landmarks.py first to generate the CSV.")
        sys.exit(1)
    
    # Step 1: Load data
    print("\n" + "=" * 60)
    print("📊 STEP 1: LOADING DATA")
    print("=" * 60)
    X, y, feature_names = load_data(CSV_PATH)
    
    # Step 2: Split data
    print("\n" + "=" * 60)
    print("✂️  STEP 2: SPLITTING DATA")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution in both sets
    )
    
    print(f"\n   Training set: {len(X_train)} samples ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"   Testing set:  {len(X_test)} samples ({TEST_SIZE*100:.0f}%)")
    
    # Step 3: Train model
    print("\n" + "=" * 60)
    print("🏋️  STEP 3: TRAINING MODEL")
    print("=" * 60)
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Step 5: Save model
    print("\n" + "=" * 60)
    print("💾 STEP 5: SAVING MODEL")
    print("=" * 60)
    save_model(model, MODEL_PATH)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n   Model: Random Forest Classifier")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Saved to: {MODEL_PATH.name}")
    print(f"\n   Ready for real-time ASL recognition! 🤟")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
