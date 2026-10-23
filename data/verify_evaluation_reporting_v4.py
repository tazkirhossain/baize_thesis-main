"""Independent arithmetic/provenance check of the final aggregate export."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "results/evaluation_reporting_v4"
report = json.loads((output / "report.json").read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


assert report["automatic_method_selection"] is False and report["best_method"] is None
assert len(report["rows"]) == 29 and report["fit_warning_count_total"] == 74
assert len(report["plots"]) == 12
for row in report["rows"]:
    m = row["metrics"]
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    div = lambda a, b: a/b if b else 0
    independent = {"adult_precision": div(tp, tp+fp), "adult_recall": div(tp, tp+fn),
                   "adult_f1": div(2*tp, 2*tp+fp+fn), "non_adult_precision": div(tn, tn+fn),
                   "non_adult_recall": div(tn, tn+fp), "non_adult_f1": div(2*tn, 2*tn+fp+fn),
                   "accuracy": (tn+tp)/(tn+fp+fn+tp),
                   "balanced_accuracy": (div(tp, tp+fn)+div(tn, tn+fp))/2}
    independent["macro_f1"] = (independent["adult_f1"]+independent["non_adult_f1"])/2
    for key, value in independent.items():
        assert math.isclose(m[key], value, abs_tol=1e-12), (key, row)
    assert m["n_records"] == tn+fp+fn+tp
    assert m["adult_count"] == fn+tp and m["non_adult_count"] == tn+fp

source = json.loads((ROOT / "data/balancing-four-methods-equal-v2/summary.json").read_text())
assert sha(ROOT / "data/balancing-four-methods-equal-v2/summary.json") == "b6866f2154285e4b74d3335c9b61e49d2a703462ebbd1217ce0ec32cf13d163a"
aliases = report["report_configuration"]["partition_aliases"]
for row in report["rows"]:
    if row["family"] != "engineered_numeric":
        continue
    selected = [run for client in source["clients"] if aliases[client["partition"]] == row["partition"]
                for run in client["runs"] if run["method"] == row["method"]]
    matrix = [[sum(run["metrics"][row["split"]]["confusion_matrix"][i][j] for run in selected)
               for j in range(2)] for i in range(2)]
    assert matrix == row["metrics"]["confusion_matrix"]
    assert row["timing"]["training_seconds_sum"] is None and row["communication"] is None

run = ROOT / "data/federated-cnn-evaluation-v4"
for number in (1, 2):
    r = json.loads((run / "server" / f"round_{number:04d}.json").read_text())
    assert r["train"]["n_records"] == 60 and r["val"]["n_records"] == 20
    assert ("test" in r) == (number == 2)
    for client in r["clients"]:
        assert client["original_train"] == client["aggregation_weight"] == 12
        assert client["used_adult"] == client["used_non_adult"]
        assert client["train_seconds"] >= 0 and client["fit_seconds"] >= client["train_seconds"]
    c = r["communication"]
    assert c["total_bytes"] == sum(c[key] for key in
        ("fit_down_bytes", "fit_up_bytes", "evaluation_down_bytes", "evaluation_up_bytes"))
    assert c["total_bytes"] > 0
for client in range(5):
    directory = run / "synthetic-clients" / f"client_{client:03d}" / "splits"
    manifest = json.loads((directory / "manifest.json").read_text())
    for split in ("train", "val", "test"):
        assert sha(directory / f"{split}.jsonl") == manifest["splits"][split]["sha256"]
for relative, expected in report["reproduction"]["report_sources_sha256"].items():
    assert sha(ROOT / relative) == expected
for name, expected in report["plots"].items():
    assert sha(output / name) == expected
    if name.endswith(".png"):
        with Image.open(output / name) as image:
            assert image.width > 1000 and image.height > 500
            image.verify()
    else:
        root = ET.parse(output / name).getroot()
        assert root.tag.endswith("svg")
assert all(not item["validation_measured"] and not item["final_test_measured"]
           for item in report["coverage"] if item["family"] != "engineered_numeric")
result = {"status": "PASS", "metric_rows_independently_checked": 29,
          "figures": 6, "png_svg_files_checked": 12, "original_numerical_summary_unchanged": True,
          "synthetic_split_files_unchanged": 15, "winner_selected": False,
          "checks": ["both-class metric arithmetic", "original confusion-count aggregation", "timing/byte accounting",
                     "source and figure hashes", "PNG/SVG integrity", "synthetic/real separation", "missing real-neural coverage"],
          "rendered_review": "six final PNG figures visually inspected; SVGs parsed; no browser/mobile or formal accessibility certification claimed",
          "report_sha256": sha(output / "report.json")}
with (output / "verification.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
