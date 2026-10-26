"""Actual client evaluation uses the same float64 score boundary as histograms."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flwr")

from scripts.federated import synthetic_clients
from src.client.flower_client import URLFlowerClient
from src.evaluation.thresholds import metrics_at_threshold, parse_histogram


class ConstantLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("logits", torch.tensor([0.0, 0.3]))

    def forward(self, x):
        return self.logits.expand(len(x), -1)


def test_client_fixed_reference_and_histogram_preserve_float64_boundary(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        cfg = json.loads((Path(__file__).parents[1]/"config/federated/character_cnn.json").read_text())
        cfg.update(progress_interval_seconds=0,
            model_selection={"source": "training_selection", "fraction": .1, "metric": "adult_f1"},
            threshold_selection={"source": "training_selection", "metric": "adult_f1",
                                 "bins": 201, "tie_break": "highest_threshold"})
        cfg["model"]["sequence_length"] = 16
        model = ConstantLogits()
        score = float(model.logits.softmax(dim=0)[1])
        cfg["threshold"] = float(np.nextafter(score, 1.0))
        # The old float32 comparison incorrectly rounds the threshold down.
        assert bool(torch.tensor(score, dtype=torch.float32) >= cfg["threshold"])
        assert score < cfg["threshold"]
        synthetic_clients(tmp_path/"clients", selection_ready=True)
        client = URLFlowerClient(0, tmp_path/"clients/client_000", cfg, models=[model])
        loss, count, metrics = client._evaluate_split("val", round_number=1)
        assert np.isfinite(loss) and count == 4
        assert [metrics[key] for key in ("tn", "fp", "fn", "tp")] == [2, 0, 2, 0]
        histogram = parse_histogram(metrics["score_histogram"], 201, expected_counts=[2, 2])
        threshold_above = np.ceil(score*200)/200
        assert metrics_at_threshold(histogram, threshold_above)["confusion_matrix"] == [[2, 0], [2, 0]]
    finally:
        torch.set_num_threads(previous_threads)
