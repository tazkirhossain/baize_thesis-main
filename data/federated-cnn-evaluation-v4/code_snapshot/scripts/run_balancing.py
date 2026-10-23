"""Run selectable client-local balancing experiments without editing source splits."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baize.data.preprocessing import _json_write, _publish_directory, _snapshot_reproduction
from baize.provenance import environment_record
from src.client.balancing_experiment import run_local_experiments
from src.data.balancing import METHODS


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile
    def write(self, text):
        self.terminal.write(text)
        return self.logfile.write(text)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT / "config/balancing/undersampling.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--client-dir",type=Path)
    group.add_argument("--partition-roots",type=Path,nargs="+")
    parser.add_argument("--methods",nargs="+",choices=[*METHODS,"all"],
                        help="omit to run the ONE configured method; all compares the four required methods")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "data"):
        parser.error("save local experiment artifacts under the ignored repository data/ directory")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiments: {output}")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    methods = args.methods
    if methods and "all" in methods:
        if methods != ["all"]:
            parser.error("all must be used alone")
        methods = ["undersampling","oversampling","smote","ensemble_undersampling"]
    jobs = []
    if args.client_dir:
        jobs.append(("single-client",args.client_dir))
    else:
        for root in args.partition_roots:
            for cid in range(5):
                jobs.append((root.name,root / f"client_{cid:03d}"))
    if len({str(path.resolve()) for _,path in jobs}) != len(jobs):
        parser.error("duplicate client directories")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preparing-balancing-",dir=output.parent) as temporary:
        ready = Path(temporary) / "ready"
        ready.mkdir()
        reproduction = _snapshot_reproduction(ready)
        all_clients = []
        with (ready / "run.log").open("w",encoding="utf-8") as logfile:
            with redirect_stdout(Tee(sys.stdout,logfile)):
                for scenario,directory in jobs:
                    manifest = json.loads((directory / "manifest.json").read_text())
                    cid = manifest["client_id"]
                    print(f"\n##### {scenario} / Client {cid+1} (storage {cid}) #####",flush=True)
                    local_config = deepcopy(cfg)
                    # Same derived seed for the same client across methods/scenarios.
                    local_config["balancing"]["seed"] = (cfg["balancing"].get("seed",42)+cid) % 2**32
                    result = run_local_experiments(directory,local_config,methods=methods)
                    result.update({"partition":scenario,"client_id":cid,"client":cid+1})
                    destination = ready / scenario / f"client_{cid:03d}"
                    destination.mkdir(parents=True)
                    _json_write(destination / "report.json",result)
                    all_clients.append(result)
                print("\nComplete: original train/validation/test files unchanged; no automatic best-method selection.",flush=True)
        summary = {"status":"PASS" if all(c["status"]=="PASS" for c in all_clients) else "COMPLETED_WITH_SKIPS",
            "scope":"local numerical-feature experiments, not FL aggregation or neural-model results",
            "model_family":"engineered_numeric", "neural_models_trained":[],
            "configuration":cfg,"methods":methods or [cfg["balancing"]["method"]],
            "environment":environment_record(),"reproduction":reproduction,"clients":all_clients}
        _json_write(ready / "summary.json",summary)
        _publish_directory(ready,output)
    print(f"Reports and full count/metric log: {output}")


if __name__ == "__main__":
    main()
