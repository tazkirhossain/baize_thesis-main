"""Create an auditable category-only skew plan from eligible-population counts."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.allocation_plans import category_counts_from_report, category_skew_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/client-datasets-v3"))
    parser.add_argument("--config-output", type=Path, default=Path("config/five_client_category_skew.json"))
    parser.add_argument("--data-output", default="data/client-category-skew-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    source = ROOT / args.source
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    counts = category_counts_from_report(report)
    plan = category_skew_plan(counts, seed=args.seed)
    config = {"strategy":"category_skew", "parameters":{"allocation_plan":plan},
        "n_clients":5, "seed":args.seed, "split_seed":args.split_seed,
        "source":str(args.source).replace("\\", "/"), "output":args.data_output}
    destination = ROOT / args.config_output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")
    print(f"Created separate category-skew plan for {sum(counts.values()):,} eligible records")
    print(f"Config: {destination}")


if __name__ == "__main__":
    main()
