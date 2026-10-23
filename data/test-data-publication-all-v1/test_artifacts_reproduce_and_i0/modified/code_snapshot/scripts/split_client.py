"""Run on one client: split its own cleaned dataset before feature extraction."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import split_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-dir", type=Path, required=True)
    args = parser.parse_args()
    report = split_client(args.client_dir)
    print(f"Client {report['client_id']} split locally:")
    for name, info in report["splits"].items():
        print(f"  {name}: {info['n_records']:,} records (adult={info['class_counts']['1']:,})")


if __name__ == "__main__":
    main()
