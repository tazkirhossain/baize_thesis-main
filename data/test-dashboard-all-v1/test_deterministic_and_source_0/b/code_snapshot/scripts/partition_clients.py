"""Prepare client datasets using a selected, reproducible partition strategy."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.partitioning import partition_prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/five_client_iid.json")
    parser.add_argument("--output", type=Path, help="new local output; existing folders are never overwritten")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    source = ROOT / cfg.pop("source")
    output = args.output or ROOT / cfg.pop("output")
    cfg.pop("output", None)
    report = partition_prepared(source, output, **cfg)
    for client in report["clients"]:
        data = client["dataset"]
        print(f"Client {client['client_id']}: Adult={data['class_counts']['1']:,}, "
              f"Non-adult={data['class_counts']['0']:,}, total={data['n_records']:,}")
    print(f"PASS: {report['partitioning']['name']}, {report['retained_records']:,} records")
    print(f"Output: {Path(output).resolve()}")
    print("Next: run scripts/split_client.py independently on each client directory.")


if __name__ == "__main__":
    main()
