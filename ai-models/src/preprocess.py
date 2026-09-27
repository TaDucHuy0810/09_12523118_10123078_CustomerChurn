"""Notebook-friendly preprocessing preview; canonical implementation is in root src/."""

from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.train_model import RANDOM_STATE, load_dataset, make_preprocessor


if __name__ == "__main__":
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    preprocessor = make_preprocessor(X)
    train_matrix = preprocessor.fit_transform(X_train)
    test_matrix = preprocessor.transform(X_test)
    print(f"Train rows: {len(X_train)}; test rows: {len(X_test)}")
    print(f"Transformed shapes: {train_matrix.shape}, {test_matrix.shape}")
    print("Preprocessor fit only on the training split.")