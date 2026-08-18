"""NLP model for operation duration prediction from theatre notes.

Turns the free-text theatre_notes column into numeric features using
TF-IDF, then trains a regression model on those text features. This
tests whether the clinical free text carries scheduling signal that
the structured columns miss.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from nbt_pipeline.preprocessing import build_preprocessed_dataset

DATA_DIR = Path("data/modelling")
TARGET = "operation_length_mins"
TEXT_COLUMN = "theatre_notes"
RANDOM_SEED = 42


def load_text_data():
    """Rebuild the dataset to recover the theatre_notes text column,
    then align it to the same train/validation split by index."""
    full = build_preprocessed_dataset()
    full = full.dropna(subset=[TARGET])

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(DATA_DIR / "validation.parquet")

    train_text = full.loc[full.index.isin(train.index), [TEXT_COLUMN, TARGET]].copy()
    val_text = full.loc[full.index.isin(validation.index), [TEXT_COLUMN, TARGET]].copy()

    train_text[TEXT_COLUMN] = train_text[TEXT_COLUMN].fillna("").astype(str)
    val_text[TEXT_COLUMN] = val_text[TEXT_COLUMN].fillna("").astype(str)

    return train_text, val_text


def run():
    train_text, val_text = load_text_data()

    print(f"Train rows with text: {(train_text[TEXT_COLUMN].str.len() > 0).sum()}")
    print(f"Validation rows with text: {(val_text[TEXT_COLUMN].str.len() > 0).sum()}")

    y_train = train_text[TARGET]
    y_val = val_text[TARGET]

    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=5,
        stop_words="english",
    )

    x_train = vectorizer.fit_transform(train_text[TEXT_COLUMN])
    x_val = vectorizer.transform(val_text[TEXT_COLUMN])

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}

    print(f"Running grid search with 5-fold cross-validation...")

    gs = GridSearchCV(
        Ridge(random_state=RANDOM_SEED), param_grid, cv=5,
        scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1,
    )
    gs.fit(x_train, y_train)

    print(f"\nBest parameters: {gs.best_params_}")
    print(f"Best CV MAE: {-gs.best_score_:.1f} mins")

    y_pred = gs.best_estimator_.predict(x_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    r2 = r2_score(y_val, y_pred)

    print(f"\nValidation results (text features only):")
    print(f"  MAE  = {mae:.1f} mins")
    print(f"  RMSE = {rmse:.1f} mins")
    print(f"  R2   = {r2:.3f}")

    hospital_mae = 40.0
    print(f"\nHospital planned time MAE: {hospital_mae:.1f} mins")
    print(f"NLP improvement: {hospital_mae - mae:.1f} mins better")

    model = gs.best_estimator_
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_
    top_long = feature_names[np.argsort(coefs)[-15:]][::-1]
    top_short = feature_names[np.argsort(coefs)[:15]]

    print(f"\nText terms most associated with LONGER operations:")
    print("  " + ", ".join(top_long))
    print(f"\nText terms most associated with SHORTER operations:")
    print("  " + ", ".join(top_short))


if __name__ == "__main__":
    run()
