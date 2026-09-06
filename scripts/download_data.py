"""Baixa o Medical Abstracts TC Corpus (treino e teste) para a pasta data/."""

import sys
from pathlib import Path

import requests

BASE_URL = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
FILES = ["medical_tc_train.csv", "medical_tc_test.csv"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_file(filename: str) -> None:
    url = f"{BASE_URL}/{filename}"
    dest = DATA_DIR / filename
    print(f"Baixando {url} -> {dest}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        download_file(filename)
    print("Download concluído.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Falha ao baixar o dataset: {exc}", file=sys.stderr)
        sys.exit(1)
