"""Collect or resume one client's website snapshot before offline FL training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.enrichment.config import resolve_enrichment
from src.data.enrichment.collect import collect_client
from src.data.enrichment.store import check_snapshot, snapshot_directory
from src.server.experiment_config import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/experiment.json')
    parser.add_argument('--client-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='pilot: process at most this many NEW rows, then resume later')
    parser.add_argument('--verify-only', action='store_true', help='verify a frozen snapshot without network access')
    args = parser.parse_args(argv)
    try:
        request = read_json(args.config)
        enrichment = resolve_enrichment(request.get('website_enrichment'), ROOT)
        if enrichment is None:
            raise ValueError('enable website_enrichment in the experiment file before collecting')
        manifest = snapshot_directory(args.client_dir, enrichment)/'manifest.json'
        frozen = manifest.is_file() and read_json(manifest).get('status') == 'complete'
        if args.verify_only or frozen:
            result = check_snapshot(args.client_dir, enrichment, verify_bytes=True)
            print(json.dumps({'status': result['status'], 'network_used': False, 'n_records': result['n_records'],
                              'input_digest': result['input_digest']}, indent=2))
        else:
            collect_client(args.client_dir, enrichment, limit=args.limit)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print('Collection interrupted. Committed records remain resumable; original splits are unchanged.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
