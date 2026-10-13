import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from spam_classifier import data, train
from spam_classifier.api import create_app

DATA_ZIP = Path(__file__).resolve().parent.parent / "data" / "raw" / "sms+spam+collection.zip"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from tests.conftest import synthetic_messages

    result = train.run(synthetic_messages(n=500), seed=7)
    out = tmp_path_factory.mktemp("artifacts")
    train.save(result, out / "model.joblib", out / "reports", dataset_sha256="0" * 64)
    return result, out


def test_run_compares_all_candidates_and_beats_baselines(trained):
    result, _ = trained
    names = [row["model"] for row in result["comparison"]]
    assert names[0] == "baseline: always ham" and len(names) == 8
    assert not result["model_name"].startswith("baseline")
    assert result["test"]["f1"] > 0.9  # synthetic data is easy; this checks the plumbing
    sizes = result["sizes"]
    assert sizes["development"] + sizes["test"] == sizes["after_dedup"]


def test_save_writes_model_and_reports(trained):
    result, out = trained
    assert (out / "model.joblib").is_file()
    report = (out / "reports" / "REPORT.md").read_text(encoding="utf-8")
    assert "(selected)" in report and "Test set (evaluated once)" in report
    metrics = json.loads((out / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_name"] == result["model_name"] and "model" not in metrics


def test_api_predicts_and_explains(trained):
    _, out = trained
    client = TestClient(create_app(out / "model.joblib"))
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}

    spam = client.post("/predict", json={"text": "urgent win cash prize claim reward"}).json()
    ham = client.post("/predict", json={"text": "see you at dinner tonight mom"}).json()
    assert spam["label"] == "spam" and spam["is_spam"] and spam["score"] >= spam["threshold"]
    assert ham["label"] == "ham" and not ham["is_spam"]
    assert spam["top_spam_features"] and spam["top_spam_features"][0]["contribution"] > 0
    assert client.get("/model").json()["model_name"]


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {"text": "x" * 1001}, {}, {"text": 5}])
def test_api_validates_input(trained, payload):
    _, out = trained
    assert TestClient(create_app(out / "model.joblib")).post("/predict", json=payload).status_code == 422


def test_api_without_model_returns_503(tmp_path):
    client = TestClient(create_app(tmp_path / "missing.joblib"))
    assert client.get("/health").json()["model_loaded"] is False
    response = client.post("/predict", json={"text": "hello"})
    assert response.status_code == 503 and "spam_classifier.train" in response.json()["detail"]


@pytest.mark.skipif(not DATA_ZIP.is_file(), reason="dataset not downloaded (python scripts/download_data.py)")
def test_real_dataset_matches_documented_counts():
    messages = data.load_zip(DATA_ZIP)
    unique = data.deduplicate(messages)
    assert len(messages) == 5574 and sum(m.label for m in messages) == 747
    assert len(unique) == 5171 and sum(m.label for m in unique) == 653
