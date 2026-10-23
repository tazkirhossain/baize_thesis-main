"""Export sanitized aggregate numerical comparison metrics from a completed local run."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.preprocessing import _file_hash
from src.evaluation.balancing_comparison import comparison_markdown, summarize_balancing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary",type=Path)
    parser.add_argument("--output",type=Path,required=True,help="new JSON result path, with matching Markdown sibling")
    args = parser.parse_args()
    markdown = args.output.with_suffix(".md")
    if args.output.suffix != ".json" or args.output.exists() or markdown.exists():
        raise ValueError("choose a new .json path; outputs are never overwritten")
    result = summarize_balancing(json.loads(args.summary.read_text(encoding="utf-8")))
    result["source_summary_sha256"] = _file_hash(args.summary)
    result["report_script_sha256"] = _file_hash(Path(__file__))
    result["aggregation_source_sha256"] = _file_hash(ROOT / "src/evaluation/balancing_comparison.py")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    text = comparison_markdown(result)
    with markdown.open("x",encoding="utf-8") as stream:
        stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
