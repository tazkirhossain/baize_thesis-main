"""Verify a complete client allocation against its audited eligible source."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.verification import verify_partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path, default=Path("data/client-datasets-v3"))
    parser.add_argument("--output", type=Path, help="new local verification report; omit for read-only check")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(f"refusing to replace verification report: {args.output}")
    result = verify_partitions(args.source, args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {result['retained_records']:,} eligible original records, exactly once across five clients")


if __name__ == "__main__":
    main()
