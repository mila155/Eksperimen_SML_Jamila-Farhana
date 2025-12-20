import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))


def load_data(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

    return processed_df


def save_processed_data(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def run_pipeline():
    input_path = os.path.join(ROOT_DIR, "breast-cancer_raw.csv")
    output_path = os.path.join(BASE_DIR, "breast-cancer_processed.csv")

    df = load_data(input_path)
    processed_df = preprocess_data(df)
    save_processed_data(processed_df, output_path)


if __name__ == "__main__":
    run_pipeline()
