"""Gera tráfego controlado para demonstrar o dashboard de observabilidade."""

import argparse
import statistics
import time
from collections import Counter

import requests


SAMPLE_REPORTS = (
    "Patient presents with acute chest pain and shortness of breath.",
    "Routine follow-up examination with stable clinical condition.",
    "Persistent headache with attention needed and neurological evaluation.",
    "Digestive discomfort without signs of acute complication.",
    "Cardiovascular emergency with severe pain and respiratory distress.",
)


def generate_traffic(
    base_url: str,
    requests_count: int,
    interval_seconds: float,
    invalid_every: int,
) -> tuple[Counter, list[float]]:
    """Envia lotes válidos e inválidos e devolve status e latências observadas."""
    status_counts: Counter = Counter()
    latencies_ms: list[float] = []

    with requests.Session() as session:
        for index in range(requests_count):
            is_invalid = invalid_every > 0 and (index + 1) % invalid_every == 0
            text = "" if is_invalid else SAMPLE_REPORTS[index % len(SAMPLE_REPORTS)]

            started_at = time.perf_counter()
            try:
                response = session.post(
                    f"{base_url.rstrip('/')}/predict",
                    json={"text": text},
                    timeout=5,
                )
                status_counts[str(response.status_code)] += 1
            except requests.RequestException as exc:
                status_counts["connection_error"] += 1
                print(f"requisição {index + 1}: erro de conexão: {exc}")
            finally:
                latencies_ms.append((time.perf_counter() - started_at) * 1000)

            if interval_seconds > 0:
                time.sleep(interval_seconds)

    return status_counts, latencies_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera requisições para popular o dashboard Grafana da Etapa 3."
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument(
        "--invalid-every",
        type=int,
        default=10,
        help="Envia texto vazio a cada N chamadas para gerar respostas 422; use 0 para desativar.",
    )
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count deve ser maior que zero")
    if args.interval < 0:
        parser.error("--interval não pode ser negativo")
    if args.invalid_every < 0:
        parser.error("--invalid-every não pode ser negativo")

    return args


def main() -> None:
    args = parse_args()
    status_counts, latencies_ms = generate_traffic(
        base_url=args.url,
        requests_count=args.count,
        interval_seconds=args.interval,
        invalid_every=args.invalid_every,
    )

    print("\nResumo da carga")
    print(f"  chamadas: {sum(status_counts.values())}")
    print(f"  status: {dict(sorted(status_counts.items()))}")
    print(f"  latência média observada pelo cliente: {statistics.mean(latencies_ms):.2f} ms")
    print(f"  maior latência observada: {max(latencies_ms):.2f} ms")


if __name__ == "__main__":
    main()
