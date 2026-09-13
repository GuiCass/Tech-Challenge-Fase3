import sys

import pytest

from scripts import generate_monitoring_traffic as traffic


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeSession:
    def __init__(self):
        self.payloads = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def post(self, _url, json, timeout):
        assert timeout == 5
        self.payloads.append(json)
        return FakeResponse(422 if json["text"] == "" else 200)


def test_generate_traffic_includes_validation_errors(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(traffic.requests, "Session", lambda: session)
    monkeypatch.setattr(traffic.time, "sleep", lambda _seconds: None)

    statuses, latencies = traffic.generate_traffic(
        base_url="http://api:8000/",
        requests_count=5,
        interval_seconds=0.01,
        invalid_every=2,
    )

    assert statuses == {"200": 3, "422": 2}
    assert len(latencies) == 5
    assert [payload["text"] == "" for payload in session.payloads] == [
        False,
        True,
        False,
        True,
        False,
    ]


def test_parse_args_rejects_non_positive_count(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate_monitoring_traffic.py", "--count", "0"])

    with pytest.raises(SystemExit):
        traffic.parse_args()
