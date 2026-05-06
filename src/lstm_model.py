"""LSTM model and sequence utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from src.data_loader import RANDOM_SEED
from src.evaluation import compute_classification_metrics


def set_tensorflow_seed() -> None:
    """Set TensorFlow-level reproducibility controls."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_lstm_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Create rolling windows ending on each row of the provided DataFrame."""
    sequences: list[np.ndarray] = []
    targets: list[int] = []
    indices: list[pd.Timestamp] = []
    values = df[feature_columns].values
    labels = df["target"].values
    for end_idx in range(window_size, len(df)):
        sequences.append(values[end_idx - window_size : end_idx])
        targets.append(labels[end_idx])
        indices.append(df.index[end_idx])
    return np.array(sequences), np.array(targets), pd.Index(indices)


def scale_lstm_frames(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Fit a scaler on train and transform all sequential inputs."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_columns])

    scaled_frames = []
    for frame in [train_df, validation_df, test_df]:
        scaled = frame.copy()
        scaled[feature_columns] = scaler.transform(frame[feature_columns])
        scaled_frames.append(scaled)
    return (*scaled_frames, scaler)


def create_lstm_model(window_size: int, feature_count: int) -> Sequential:
    """Construct the required LSTM architecture."""
    model = Sequential(
        [
            LSTM(32, input_shape=(window_size, feature_count)),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
) -> dict[str, object]:
    """Train the LSTM using rolling windows and chronological validation."""
    set_tensorflow_seed()
    scaled_train, scaled_validation, scaled_test, scaler = scale_lstm_frames(
        train_df,
        validation_df,
        test_df,
        feature_columns,
    )

    validation_with_context = pd.concat(
        [scaled_train.tail(window_size), scaled_validation],
        axis=0,
    )
    test_with_context = pd.concat(
        [scaled_validation.tail(window_size), scaled_test],
        axis=0,
    )

    x_train, y_train, train_index = build_lstm_sequences(scaled_train, feature_columns, window_size)
    x_validation, y_validation, validation_index = build_lstm_sequences(
        validation_with_context,
        feature_columns,
        window_size,
    )
    x_test, y_test, test_index = build_lstm_sequences(test_with_context, feature_columns, window_size)

    model = create_lstm_model(window_size, len(feature_columns))
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=20,
        batch_size=32,
        verbose=0,
    )

    validation_probabilities = model.predict(x_validation, verbose=0).ravel()
    validation_predictions = (validation_probabilities >= 0.5).astype(int)
    test_probabilities = model.predict(x_test, verbose=0).ravel()
    test_predictions = (test_probabilities >= 0.5).astype(int)

    return {
        "model": model,
        "scaler": scaler,
        "history": history,
        "train_index": train_index,
        "validation_index": validation_index,
        "test_index": test_index,
        "validation_probabilities": validation_probabilities,
        "validation_predictions": validation_predictions,
        "validation_metrics": compute_classification_metrics(y_validation, validation_predictions),
        "test_probabilities": test_probabilities,
        "test_predictions": test_predictions,
        "test_metrics": compute_classification_metrics(y_test, test_predictions),
    }


def save_training_curve_plot(history, output_path: Path) -> None:
    """Save loss and accuracy training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set_title("LSTM Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Validation")
    axes[1].set_title("LSTM Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

