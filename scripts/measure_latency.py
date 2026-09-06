"""Mede a latência baseline do endpoint /predict rodando localmente/em container.

Uso:
    python scripts/measure_latency.py --url http://localhost:8000 --n 200
"""

import argparse
import statistics
import time

import requests

SAMPLE_TEXTS = [
    "Patient presents with acute chest pain and shortness of breath.",
    "Routine follow-up visit, patient feels well, no new symptoms reported.",
    "Severe abdominal pain with signs of internal bleeding, requires immediate attention.",
    "Mild headache and fatigue reported during annual checkup.",
    "Sudden loss of consciousness and seizure activity observed.",
]


def percentile(data: list[float], p: float) -> float:
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)


def run(url: str, n: int) -> None:
    endpoint = f"{url.rstrip('/')}/predict"
    latencies_ms = []

    # warm-up
    requests.post(endpoint, json={"text": SAMPLE_TEXTS[0]}, timeout=10)

    for i in range(n):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        start = time.perf_counter()
        response = requests.post(endpoint, json={"text": text}, timeout=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        latencies_ms.append(elapsed_ms)

    print(f"Requisições: {n}")
    print(f"Média: {statistics.mean(latencies_ms):.2f} ms")
    print(f"p50: {percentile(latencies_ms, 50):.2f} ms")
    print(f"p95: {percentile(latencies_ms, 95):.2f} ms")
    print(f"p99: {percentile(latencies_ms, 99):.2f} ms")
    print(f"min/max: {min(latencies_ms):.2f} / {max(latencies_ms):.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    run(args.url, args.n)
