"""Check data comparability and export aggregate partition distributions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.comparison import compare_partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_partitions(args.roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves prior comparison results.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"PASS: {len(result['runs'])} comparable data preparations")
    print(result["scope"])


if __name__ == "__main__":
    main()
