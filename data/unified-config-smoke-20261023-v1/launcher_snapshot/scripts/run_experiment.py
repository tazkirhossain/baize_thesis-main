"""One configuration, one launcher for the existing Flower neural pipeline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.server.experiment_config import check_plan, read_json, resolve_experiment


def project_path(value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (ROOT/value).resolve()


def output_path(plan, requested=None):
    cfg = plan.configuration
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    name = f"{cfg['experiment']['partition_strategy']}_{cfg['balancing']['method']}_{cfg['model']['family']}_{stamp}_{uuid.uuid4().hex[:6]}"
    path = project_path(requested) if requested else ROOT/'data'/name
    if path.parent != (ROOT/'data').resolve() or path.exists():
        raise ValueError('output must be a new immediate child of ignored data/ (for dashboard discovery)')
    return path


def training_command(plan, resolved_file, args, output):
    command = [sys.executable, str(ROOT/'scripts/federated.py'), args.mode,
               '--config', str(resolved_file), '--threads', str(plan.threads)]
    if args.mode != 'client':
        command += ['--output', str(output)]
    if args.mode in ('local', 'client'):
        command += ['--device', plan.device]
    if args.mode == 'local':
        command += ['--partition-root', str(plan.partition_root)]
    if args.mode == 'client':
        directory = project_path(args.client_dir) if args.client_dir else plan.partition_root/f'client_{args.client_id:03d}'
        command += ['--client-id', str(args.client_id), '--client-dir', str(directory)]
    if args.mode in ('server', 'client'):
        command += ['--address', args.address]
        if args.ca:
            command += ['--ca', str(project_path(args.ca))]
    if args.mode == 'server':
        for key in ('cert', 'key'):
            if getattr(args, key):
                command += [f'--{key}', str(project_path(getattr(args, key)))]
    return command


def save_launcher_record(output, request, plan, mode):
    (output/'experiment-request.json').write_text(json.dumps(request, indent=2)+'\n', encoding='utf-8')
    sources = {}
    for relative in ('scripts/run_experiment.py', 'src/server/experiment_config.py'):
        target = output/'launcher_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        sources[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    (output/'launcher_reproduction.json').write_text(json.dumps({
        'mode': mode, 'request_sha256': hashlib.sha256((output/'experiment-request.json').read_bytes()).hexdigest(),
        'model_defaults': plan.default_file.relative_to(ROOT).as_posix(),
        'model_defaults_sha256': hashlib.sha256(plan.default_file.read_bytes()).hexdigest(),
        'source_sha256': sources,
        'effective_configuration': 'experiment.json',
        'partition_root': str(plan.partition_root) if mode == 'local' else None,
    }, indent=2)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/experiment.json')
    parser.add_argument('--mode', choices=('local', 'smoke', 'server', 'client'), default='local')
    parser.add_argument('--check', action='store_true', help='validate and print settings without training or creating run artifacts')
    parser.add_argument('--output', help='optional fresh data/RUN folder; otherwise generated automatically')
    parser.add_argument('--client-id', type=int)
    parser.add_argument('--client-dir')
    parser.add_argument('--address', default='127.0.0.1:8080')
    parser.add_argument('--ca')
    parser.add_argument('--cert')
    parser.add_argument('--key')
    args = parser.parse_args(argv)
    try:
        request = read_json(project_path(args.config))
        plan = resolve_experiment(request, ROOT)
        if args.mode == 'client' and args.output:
            raise ValueError('client mode does not write a central run output')
        if args.mode in ('server', 'client'):
            local = args.address.startswith(('127.0.0.1:', 'localhost:', '[::1]:'))
            if not local and not args.ca:
                raise ValueError('remote connections require TLS --ca')
            if args.mode == 'server' and not (bool(args.ca) == bool(args.cert) == bool(args.key)):
                raise ValueError('server TLS requires --ca, --cert and --key together')
        directory = project_path(args.client_dir) if args.client_dir else None
        if args.mode in ('local', 'client') and plan.device != 'cpu':
            import torch
            device = torch.device(plan.device)
            if device.type == 'cuda' and (not torch.cuda.is_available() or
                    (device.index is not None and device.index >= torch.cuda.device_count())):
                raise ValueError(f'requested {plan.device} is unavailable; choose cpu or install/use a compatible CUDA environment')
            if device.type == 'mps' and not torch.backends.mps.is_available():
                raise ValueError('requested mps device is unavailable')
        check_plan(plan, mode=args.mode, client_id=args.client_id, client_directory=directory)
        output = None if args.mode == 'client' else output_path(plan, args.output)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    cfg = plan.configuration
    print(f"Model: {cfg['model']['family']} | Distribution: {cfg['experiment']['partition_strategy']} | Balancing: {cfg['balancing']['method']}", flush=True)
    print(f"Protocol: {request['training_protocol']} | Rounds: {cfg['rounds']} | Local epochs: {cfg['local_epochs']} | Batch: {cfg['batch_size']} | LR: {cfg['learning_rate']}", flush=True)
    print(f"Device: {plan.device} | Threads: {plan.threads} | FedAvg weights: {cfg['aggregation_weight']}", flush=True)
    if args.mode == 'smoke':
        print('SMOKE overrides: synthetic fixtures, unknown distribution, rounds=2, batch_size=8, sequence_length=32, CPU. Configured local_epochs is retained.', flush=True)
    elif args.mode in ('local', 'client'):
        print(f'Assigned data: {directory or plan.partition_root}', flush=True)
    if args.check:
        print('PASS: configuration and required local metadata/files are available. No training or full data-content audit was performed.', flush=True)
        print(json.dumps(cfg, indent=2), flush=True)
        return 0
    if output:
        print(f'New run output: {output}', flush=True)
    (ROOT/'data').mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='launcher-config-', dir=ROOT/'data') as temporary:
        resolved_file = Path(temporary)/'resolved.json'
        resolved_file.write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
        result = subprocess.run(training_command(plan, resolved_file, args, output), cwd=ROOT)
    if output and output.is_dir():
        save_launcher_record(output, request, plan, args.mode)
    if result.returncode:
        print('Training endpoint failed. No successful training result is claimed.', file=sys.stderr)
        return result.returncode
    if output and plan.generate_plots:
        # The declared ratio configures reporting before looking at any scores.
        settings = read_json(ROOT/'config/reporting/comparison.json')
        settings['expected_adult_to_non_adult'] = cfg['balancing']['adult_to_non_adult']
        settings_file = output/'report-settings.json'
        settings_file.write_text(json.dumps(settings, indent=2)+'\n', encoding='utf-8')
        report = subprocess.run([sys.executable, str(ROOT/'scripts/report_evaluation.py'),
            '--config', str(settings_file), '--federated-run', str(output/'server'),
            '--output', str(output/'report')], cwd=ROOT)
        if report.returncode:
            print(f'Training completed; plotting failed. Saved training results remain in {output}.', file=sys.stderr)
            return report.returncode
        print(f'Matplotlib figures and measured metrics: {output / "report"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
