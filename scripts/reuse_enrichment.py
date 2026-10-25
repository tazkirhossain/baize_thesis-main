"""Owner-operated OFFLINE feature provisioning for matched cross-scenario studies.

This is never invoked by the Flower server. Runtime clients read only their own
destination cache after provisioning. Source snapshots must already be frozen.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.enrichment.config import resolve_enrichment
from src.data.enrichment.reuse import reuse_client_snapshot
from src.data.enrichment.store import check_snapshot, snapshot_directory
from src.server.experiment_config import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/experiment_website.json')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--target-client-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        enrichment = resolve_enrichment(read_json(args.config).get('website_enrichment'), ROOT)
        if not enrichment:
            raise ValueError('enable website_enrichment first')
        manifest = snapshot_directory(args.target_client_dir, enrichment)/'manifest.json'
        if manifest.exists() and read_json(manifest).get('status') == 'complete':
            result = check_snapshot(args.target_client_dir, enrichment, verify_bytes=True)
        else:
            result = reuse_client_snapshot(args.source_root, args.target_client_dir, enrichment)
        print(json.dumps({'status':result['status'], 'n_records':result['n_records'],
                          'network_used':False, 'input_digest':result['input_digest']}, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print('Reuse interrupted; committed target records remain resumable.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
