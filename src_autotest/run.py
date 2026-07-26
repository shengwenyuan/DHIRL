#!/usr/bin/env python
"""
Autotest runner — reads a YAML config and launches train_labyrinth_b.py
experiments with full logging.

Usage (from DHIRL root):
    python -m src_autotest.run src_autotest/configs/test0310.yaml
    python -m src_autotest.run src_autotest/configs/test0310.yaml --groups model_comparison
"""

import os
import sys
import yaml
import subprocess
import datetime
import argparse
import hashlib
import json
import shutil

TRAIN_MODULES = {
    'labyrinth': 'src_autotest.train_labyrinth_b',
    'gridworld': 'src_autotest.train_gridworld',
}

PARAM_KEYS = [
    'll_filename', 'output_dir', 'group_id', 'data_dir',
    'num_repeats', 'num_latents', 'rand_seed',
    'model_type', 'hidden_dim', 'rnn_hidden_dim', 'num_layers', 'dropout', 'nhead', 'lr',
    'reg_type', 'reg_weight', 'kl_weight',
    'num_epochs', 'loss_threshold', 'max_iterations', 'gate_mode',
    'fold_idx', 'max_trajs', 'paired_fold_seeds', 'p0_artifacts',
    'retrospective_score_mode',
]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_command(params):
    task = params.get('task', 'labyrinth')
    if task not in TRAIN_MODULES:
        raise ValueError(f'Unknown task={task!r}; expected one of {tuple(TRAIN_MODULES)}')
    cmd = [sys.executable, '-m', TRAIN_MODULES[task]]
    for key in PARAM_KEYS:
        if key in params:
            cmd += [f'--{key}', str(params[key])]
    return cmd


def label_from_overrides(exp, defaults):
    """Derive a short human-readable label from the keys that differ from defaults."""
    parts = []
    for k, v in exp.items():
        if k in defaults and defaults[k] == v:
            continue
        parts.append(f'{k}={v}')
    return ', '.join(parts) if parts else '(defaults)'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fin:
        for block in iter(lambda: fin.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_tree_sha256(root):
    manifest_path = os.path.join(root, 'SHA256SUMS')
    paths = []
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(directory, filename)
            if os.path.abspath(path) != os.path.abspath(manifest_path):
                paths.append(path)
    with open(manifest_path, 'w') as fout:
        for path in sorted(paths):
            relative = os.path.relpath(path, root)
            fout.write(f'{sha256_file(path)}  {relative}\n')


def normalized_requested(defaults, groups):
    return any(
        {**defaults, **experiment}.get('retrospective_score_mode')
        == 'action_normalized'
        for group in groups.values()
        for experiment in group.get('experiments', [])
    )


def run_one(cmd, log_path, group_name, label, env):
    start = datetime.datetime.now()
    with open(log_path, 'w') as lf:
        lf.write(f'group  : {group_name}\n')
        lf.write(f'label  : {label}\n')
        lf.write(f'command: {" ".join(cmd)}\n')
        lf.write(f'started: {start.isoformat()}\n')
        lf.write('=' * 72 + '\n\n')
        lf.flush()

        result = subprocess.run(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env
        )

        end = datetime.datetime.now()
        elapsed = end - start
        lf.write('\n' + '=' * 72 + '\n')
        lf.write(f'finished : {end.isoformat()}\n')
        lf.write(f'elapsed  : {elapsed}\n')
        lf.write(f'exit_code: {result.returncode}\n')

    return result.returncode, elapsed


def finalize_experiment_provenance(
    row, config_path, config_sha256, runner_manifest_path, runner_manifest_sha256
):
    experiment_dir = row['experiment_dir']
    manifest_path = os.path.join(experiment_dir, 'run_manifest.json')
    with open(manifest_path) as fin:
        manifest = json.load(fin)
    provenance = manifest.get('provenance', {})
    expected = {
        'config_path': config_path,
        'config_sha256': config_sha256,
        'runner_log_path': row['log_path'],
        'command': row['command'],
    }
    differences = [
        key for key, value in expected.items()
        if provenance.get(key) != value
    ]
    if differences:
        raise RuntimeError(
            f'{experiment_dir}: trainer provenance mismatch for {differences}.'
        )
    provenance.update({
        'runner_log_sha256': sha256_file(row['log_path']),
        'runner_manifest_path': runner_manifest_path,
        'runner_manifest_sha256': runner_manifest_sha256,
    })
    manifest['provenance'] = provenance
    with open(manifest_path, 'w') as fout:
        json.dump(manifest, fout, indent=2)
        fout.write('\n')
    write_tree_sha256(experiment_dir)


def main():
    parser = argparse.ArgumentParser(description='Autotest runner')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    parser.add_argument('--groups', type=str, nargs='*', default=None,
                        help='Run only the listed groups (default: all)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Override log directory')
    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get('defaults', {})
    groups = cfg.get('groups', {})

    if args.groups:
        groups = {k: v for k, v in groups.items() if k in args.groups}

    normalized_run = normalized_requested(defaults, groups)
    provenance_cfg = cfg.get('provenance', {})
    config_path = os.path.abspath(args.config)
    config_sha256 = sha256_file(config_path)
    git_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True
    ).strip()
    git_branch = subprocess.check_output(
        ['git', 'branch', '--show-current'], text=True
    ).strip()
    git_dirty = bool(subprocess.check_output(
        ['git', 'status', '--short'], text=True
    ).strip())
    baseline_tag = provenance_cfg.get('baseline_tag')
    baseline_ancestor = None
    if normalized_run:
        if provenance_cfg.get('purpose') not in {
            'formal_rebuttal', 'smoke_validation'
        }:
            raise ValueError(
                'Normalized configs require provenance.purpose to be '
                'formal_rebuttal or smoke_validation.'
            )
        if not baseline_tag:
            raise ValueError(
                'Normalized configs require provenance.baseline_tag.'
            )
        if git_branch != '26r':
            raise RuntimeError(
                f'Normalized runs require branch 26r, found {git_branch!r}.'
            )
        if git_dirty:
            raise RuntimeError(
                'Refusing normalized run from a dirty worktree.'
            )
        baseline_ancestor = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', baseline_tag, git_commit],
            check=False,
        ).returncode == 0
        if not baseline_ancestor:
            raise RuntimeError(
                f'Baseline tag {baseline_tag!r} is not an ancestor of '
                f'{git_commit}.'
            )

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root = args.log_dir or os.path.join('src_autotest', 'logs', timestamp)
    if (
        normalized_run
        and os.path.isdir(log_root)
        and os.listdir(log_root)
    ):
        raise FileExistsError(
            f'Refusing to reuse non-empty normalized-run log directory: '
            f'{log_root}'
        )
    os.makedirs(log_root, exist_ok=True)
    shutil.copy2(args.config, os.path.join(log_root, 'config.yaml'))
    with open(os.path.join(log_root, 'command.txt'), 'w') as fout:
        fout.write(' '.join(sys.argv) + '\n')
    with open(os.path.join(log_root, 'git_commit.txt'), 'w') as fout:
        fout.write(git_commit + '\n')

    summary_rows = []
    failed_exit_code = 0
    runner_manifest_path = os.path.abspath(
        os.path.join(log_root, 'run_manifest.json')
    )

    for group_name, group_cfg in groups.items():
        description = group_cfg.get('description', '')
        experiments = group_cfg.get('experiments', [])
        gid = group_cfg.get('id', group_name)

        group_log_dir = os.path.join(log_root, gid)
        os.makedirs(group_log_dir, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'  Group: {gid}  ({len(experiments)} experiments)')
        if description:
            print(f'  {description}')
        print(f'{"="*60}')

        for idx, exp in enumerate(experiments):
            exp = dict(exp)
            eid = exp.pop('id', f'E{idx:02d}')
            params = {**defaults, **exp}
            params['group_id'] = f'{gid}/{eid}'

            if 'output_dir' in params:
                params['output_dir'] = os.path.join(params['output_dir'], timestamp)

            label = label_from_overrides(exp, defaults)
            cmd = build_command(params)
            log_path = os.path.abspath(
                os.path.join(group_log_dir, f'{eid}.log')
            )
            experiment_dir = (
                os.path.abspath(
                    os.path.join(params['output_dir'], params['group_id'])
                )
                if 'output_dir' in params
                else None
            )
            env = os.environ.copy()
            if normalized_run:
                env.update({
                    'PRISM_RUN_PURPOSE': provenance_cfg['purpose'],
                    'PRISM_BASELINE_TAG': baseline_tag,
                    'PRISM_RUN_CONFIG_PATH': config_path,
                    'PRISM_RUN_CONFIG_SHA256': config_sha256,
                    'PRISM_RUNNER_LOG_PATH': log_path,
                    'PRISM_RUNNER_MANIFEST_PATH': runner_manifest_path,
                    'PRISM_RUN_COMMAND_JSON': json.dumps(cmd),
                })

            print(f'\n  >> [{gid}/{eid}] {label}')
            print(f'     log: {log_path}')

            return_code, elapsed = run_one(
                cmd, log_path, gid, label, env
            )
            current_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], text=True
            ).strip()
            current_dirty = bool(subprocess.check_output(
                ['git', 'status', '--short'], text=True
            ).strip())
            provenance_drift = (
                normalized_run
                and (current_commit != git_commit or current_dirty)
            )
            if provenance_drift and return_code == 0:
                return_code = 97
            status = 'OK' if return_code == 0 else f'FAIL({return_code})'

            print(f'     {status}  ({elapsed})')
            summary_rows.append({
                'tag': f'{gid}/{eid}',
                'label': label,
                'command': cmd,
                'status': status,
                'exit_code': return_code,
                'elapsed': str(elapsed),
                'log_path': log_path,
                'experiment_dir': experiment_dir,
                'provenance_drift': provenance_drift,
            })
            if return_code != 0:
                failed_exit_code = return_code
                break
        if failed_exit_code:
            break

    with open(runner_manifest_path, 'w') as fout:
        json.dump({
            'timestamp': timestamp,
            'purpose': provenance_cfg.get('purpose'),
            'config': config_path,
            'config_sha256': config_sha256,
            'git_commit': git_commit,
            'git_branch': git_branch,
            'git_dirty': git_dirty,
            'baseline_tag': baseline_tag,
            'baseline_ancestor': baseline_ancestor,
            'status': 'failed' if failed_exit_code else 'complete',
            'jobs': summary_rows,
        }, fout, indent=2)
        fout.write('\n')

    if normalized_run and not failed_exit_code:
        runner_manifest_sha256 = sha256_file(runner_manifest_path)
        for row in summary_rows:
            finalize_experiment_provenance(
                row,
                config_path,
                config_sha256,
                runner_manifest_path,
                runner_manifest_sha256,
            )

    summary_path = os.path.join(log_root, 'summary.txt')
    with open(summary_path, 'w') as sf:
        sf.write(f'Autotest Summary  {timestamp}\n')
        sf.write('=' * 72 + '\n')
        for row in summary_rows:
            sf.write(f"{row['tag']:<12s}  {row['label']:<40s}  "
                     f"{row['status']:<12s}  {row['elapsed']}\n")

    print(f'\n\nAll done.  Logs: {log_root}')
    print(f'Summary:   {summary_path}')
    write_tree_sha256(log_root)
    if failed_exit_code:
        raise SystemExit(failed_exit_code)


if __name__ == '__main__':
    main()
