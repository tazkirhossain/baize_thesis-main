"""Prepare client-owned FL datasets from a local CSV; never fetch URLs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import PreprocessingConfig, prepare_clients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/url.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/client-datasets-v3")
    parser.add_argument("--header", choices=("auto", "present", "absent"), default="auto")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fractions", nargs=3, type=float, default=(0.70, 0.15, 0.15),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--min-per-class", type=int, default=10)
    parser.add_argument("--features", type=int, default=2**16)
    args = parser.parse_args()
    print(f"Preparing {args.clients} client datasets from {args.input.resolve()}", flush=True)
    print("Validating, deduplicating and assigning whole domains; a full corpus may take several minutes.",
          flush=True)
    report = prepare_clients(args.input, args.output, PreprocessingConfig(
        n_clients=args.clients, seed=args.seed, fractions=tuple(args.fractions),
        min_per_class=args.min_per_class, n_features=args.features, header=args.header))
    print(f"PASS: {report['retained_records']:,} records across {args.clients} clients")
    for client in report["clients"]:
        dataset = client["dataset"]
        print(f"  client {client['client_id']}: {dataset['n_records']:,} unsplit records "
              f"(adult={dataset['class_counts']['1']:,})")
    print(f"Artifacts: {args.output.resolve()}")
    print("Next: each client runs scripts/split_client.py --client-dir <its-directory>")


if __name__ == "__main__":
    main()
