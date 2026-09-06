"""Carregamento do Medical Abstracts TC Corpus e mapeamento para níveis de urgência.

O dataset original classifica categoria de doença (condition_label 1-5), não
urgência clínica. Para este exercício de triagem, mapeamos cada categoria para
um nível de urgência (normal/atenção/urgente) — ver decisão documentada no
README. Isso é uma simplificação didática, não um mapeamento clinicamente validado.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

CONDITION_LABELS = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

URGENCY_MAP = {
    1: "atenção",
    2: "normal",
    3: "atenção",
    4: "urgente",
    5: "normal",
}

URGENCY_LEVELS = ["normal", "atenção", "urgente"]


def _load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode `python scripts/download_data.py` primeiro."
        )
    df = pd.read_csv(path)
    df = df.rename(columns={"medical_abstract": "text", "condition_label": "condition_label"})
    df["urgency"] = df["condition_label"].map(URGENCY_MAP)
    return df[["text", "condition_label", "urgency"]]


def load_train_df() -> pd.DataFrame:
    return _load_csv("medical_tc_train.csv")


def load_test_df() -> pd.DataFrame:
    return _load_csv("medical_tc_test.csv")


if __name__ == "__main__":
    train_df = load_train_df()
    test_df = load_test_df()
    print(f"Treino: {len(train_df)} amostras | Teste: {len(test_df)} amostras")
    print("Distribuição de urgência (treino):")
    print(train_df["urgency"].value_counts())
