"""Create the reproducible five-client IID baseline from audited cleaned data."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.iid_partitioning import partition_prepared_iid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/five_client_iid.json")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    source = args.source or ROOT / cfg.pop("source")
    output = args.output or ROOT / cfg.pop("output")
    cfg.pop("source", None)
    cfg.pop("output", None)
    if cfg.pop("strategy", "iid") != "iid" or cfg.pop("parameters", {}):
        parser.error("partition_iid.py accepts only IID; use partition_clients.py for other strategies")
    if args.seed is not None:
        cfg["seed"] = args.seed
    report = partition_prepared_iid(source, output, **cfg)
    for client in report["clients"]:
        info = client["dataset"]
        print(f"Client {client['client_id']}: Adult={info['class_counts']['1']:,}, "
              f"Non-adult={info['class_counts']['0']:,}, total={info['n_records']:,}")
    print(f"PASS: {report['retained_records']:,} unique records across exactly five clients")
    print(f"Output: {Path(output).resolve()}")


if __name__ == "__main__":
    main()
