import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from env.gridworld import GridWorld
from src.algorithms import PGIAVI


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fin:
        for block in iter(lambda: fin.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pad_sequences(sequences, fill_value=0.0, dtype=np.float32):
    max_len = max(len(sequence) for sequence in sequences)
    trailing_shape = np.asarray(sequences[0]).shape[1:]
    padded = np.full(
        (len(sequences), max_len, *trailing_shape),
        fill_value,
        dtype=dtype,
    )
    mask = np.zeros((len(sequences), max_len), dtype=bool)
    for idx, sequence in enumerate(sequences):
        sequence = np.asarray(sequence)
        padded[idx, :len(sequence)] = sequence
        mask[idx, :len(sequence)] = True
    return padded, mask


def save_checkpoint(model, path):
    state_dict = {
        name: value.detach().cpu()
        for name, value in model.target_intention_net.state_dict().items()
    }
    torch.save(state_dict, path)


def write_sha256_manifest(directory):
    filenames = sorted(
        name
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name)) and name != 'SHA256SUMS'
    )
    with open(os.path.join(directory, 'SHA256SUMS'), 'w') as fout:
        for filename in filenames:
            fout.write(f'{sha256_file(os.path.join(directory, filename))}  {filename}\n')


def normalized_run_provenance(fold_idx):
    env_keys = {
        'purpose': 'PRISM_RUN_PURPOSE',
        'baseline_tag': 'PRISM_BASELINE_TAG',
        'config_path': 'PRISM_RUN_CONFIG_PATH',
        'config_sha256': 'PRISM_RUN_CONFIG_SHA256',
        'runner_log_path': 'PRISM_RUNNER_LOG_PATH',
        'runner_manifest_path': 'PRISM_RUNNER_MANIFEST_PATH',
        'command_json': 'PRISM_RUN_COMMAND_JSON',
    }
    context = {
        key: os.environ.get(env_name)
        for key, env_name in env_keys.items()
    }
    missing = [key for key, value in context.items() if not value]
    if fold_idx < 0 and missing:
        raise RuntimeError(
            f'Formal normalized run is missing runner provenance: {missing}.'
        )

    git_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True
    ).strip()
    git_branch = subprocess.check_output(
        ['git', 'branch', '--show-current'], text=True
    ).strip()
    git_dirty = bool(subprocess.check_output(
        ['git', 'status', '--short'], text=True
    ).strip())
    provenance = {
        'purpose': context['purpose'] or 'manual_validation',
        'git_commit': git_commit,
        'git_branch': git_branch,
        'git_dirty': git_dirty,
        'python_executable': sys.executable,
        'baseline_tag': context['baseline_tag'],
        'baseline_ancestor': None,
        'config_path': context['config_path'],
        'config_sha256': context['config_sha256'],
        'runner_log_path': context['runner_log_path'],
        'runner_manifest_path': context['runner_manifest_path'],
        'command': (
            json.loads(context['command_json'])
            if context['command_json']
            else [sys.executable, *sys.argv]
        ),
    }
    if context['config_path']:
        if sha256_file(context['config_path']) != context['config_sha256']:
            raise RuntimeError('Runner config hash does not match its file.')
    if context['baseline_tag']:
        provenance['baseline_ancestor'] = subprocess.run(
            [
                'git', 'merge-base', '--is-ancestor',
                context['baseline_tag'], git_commit,
            ],
            check=False,
        ).returncode == 0
    if fold_idx < 0:
        if context['purpose'] != 'formal_rebuttal':
            raise RuntimeError(
                'Full normalized runs require purpose=formal_rebuttal.'
            )
        if (
            git_branch != '26r'
            or git_dirty
            or provenance['baseline_ancestor'] is not True
        ):
            raise RuntimeError(
                'Formal normalized run requires clean 26r with the baseline '
                'tag as an ancestor.'
            )
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ll_filename', type=str, default='metrics.csv')
    parser.add_argument('--output_dir', type=str, default='src_autotest/outputs_26r')
    parser.add_argument('--group_id', type=str, default='default')
    parser.add_argument('--data_dir', type=str, default='data/gridworld')
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--num_latents', type=int, default=2)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--gate_mode', type=str, default='retrospective',
                        choices=['retrospective', 'causal', 'state_only'])
    parser.add_argument(
        '--retrospective_score_mode',
        type=str,
        default='legacy',
        choices=['legacy', 'action_normalized'],
    )
    parser.add_argument('--loss_threshold', type=float, default=1e-3)
    parser.add_argument('--max_iterations', type=int, default=100)
    parser.add_argument('--fold_idx', type=int, default=-1,
                        help='Fold to run for smoke testing. -1 runs all folds.')
    parser.add_argument('--max_trajs', type=int, default=0,
                        help='Prefix size for smoke testing. 0 uses all trajectories.')
    args = parser.parse_args()

    if args.num_repeats != 1:
        raise ValueError('P0 Gridworld runs require num_repeats=1; use separate rand_seed experiments.')
    if (
        args.retrospective_score_mode == 'action_normalized'
        and args.gate_mode != 'retrospective'
    ):
        raise ValueError(
            'action_normalized retrospective scoring requires gate_mode=retrospective.'
        )

    num_folds = 5
    if args.fold_idx < -1 or args.fold_idx >= num_folds:
        raise ValueError(f'fold_idx must be -1 or in [0, {num_folds - 1}].')
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    envr = GridWorld()
    expected_p_shape = (envr.num_states, envr.num_states, envr.num_actions)
    if envr.P.shape != expected_p_shape:
        raise ValueError(f'Expected GridWorld.P shape {expected_p_shape}, got {envr.P.shape}')
    if not np.allclose(envr.P.sum(axis=1), 1.0):
        raise ValueError('GridWorld.P must sum to one over next states.')
    trajs_path = os.path.join(args.data_dir, 'trajs_frustration.json')
    latents_path = os.path.join(args.data_dir, 'latents_frustration.json')
    with open(trajs_path) as fin:
        trajs = json.load(fin)
    with open(latents_path) as fin:
        latents = json.load(fin)
    if len(trajs) != len(latents):
        raise ValueError('Gridworld trajectories and latent labels have different counts.')
    if not all(len(traj) == len(labels) for traj, labels in zip(trajs, latents)):
        raise ValueError('Gridworld latent labels are not step-aligned with trajectories.')
    total_trajs = len(trajs)
    total_steps = sum(len(traj) for traj in trajs)
    if total_trajs != 1024 or total_steps != 8036:
        raise ValueError(
            f'Expected canonical Gridworld size 1024/8036, got {total_trajs}/{total_steps}.'
        )
    num_trajs = total_trajs if args.max_trajs <= 0 else min(args.max_trajs, total_trajs)
    if num_trajs < num_folds:
        raise ValueError(f'Need at least {num_folds} trajectories, got {num_trajs}.')
    trajs = trajs[:num_trajs]
    latents = latents[:num_trajs]

    run_dir = os.path.join(args.output_dir, args.group_id)
    if (
        args.retrospective_score_mode == 'action_normalized'
        and os.path.isdir(run_dir)
        and os.listdir(run_dir)
    ):
        raise FileExistsError(
            f'Refusing to reuse non-empty normalized-score run directory: {run_dir}'
        )
    os.makedirs(run_dir, exist_ok=True)
    resolved_args = vars(args).copy()
    if args.retrospective_score_mode == 'legacy':
        resolved_args.pop('retrospective_score_mode')
    manifest = {
        'domain': 'gridworld',
        'gate_mode': args.gate_mode,
        'model_type': 'IntentionRNN',
        'num_latents': args.num_latents,
        'num_states': envr.num_states,
        'num_actions': envr.num_actions,
        'rand_seed': args.rand_seed,
        'num_repeats': args.num_repeats,
        'num_folds': num_folds,
        'fold_idx': args.fold_idx,
        'fold_random_state': 10015,
        'fold_seed_rule': 'rand_seed + 1009 * fold_idx',
        'fold_seeds': [args.rand_seed + 1009 * idx for idx in range(num_folds)],
        'transition_shape': list(envr.P.shape),
        'num_trajs_total': total_trajs,
        'num_trajs_used': num_trajs,
        'num_steps_total': total_steps,
        'score_type': (
            'retrospective_compatibility_score'
            if args.gate_mode == 'retrospective'
            else 'predictive_action_log_likelihood'
        ),
        'status': 'running',
        'trajectory_path': os.path.abspath(trajs_path),
        'latent_path': os.path.abspath(latents_path),
        'trajectory_bytes': os.path.getsize(trajs_path),
        'latent_bytes': os.path.getsize(latents_path),
        'trajectory_sha256': sha256_file(trajs_path),
        'latent_sha256': sha256_file(latents_path),
        'resolved_args': resolved_args,
        'python': platform.python_version(),
        'torch': torch.__version__,
        'device': 'cpu',
    }
    normalized_mode = args.retrospective_score_mode == 'action_normalized'
    provenance = (
        normalized_run_provenance(args.fold_idx)
        if normalized_mode
        else None
    )
    if normalized_mode:
        manifest.update({
            'artifact_schema': 'p0_sequence_artifacts_v2',
            'checkpoint_provenance': 'recreated_submitted_configuration',
            'training_objective': 'legacy_retrospective_compatibility',
            'retrospective_score_mode': args.retrospective_score_mode,
            'candidate_context_rule': (
                'observed_prefix_fixed_current_action_enumerated'
            ),
            'policy_epsilon': 1e-8,
            'candidate_policy_convention': (
                'row_normalized_softmax_plus_epsilon'
            ),
            'legacy_policy_log_scale': float(
                np.log1p(envr.num_actions * 1e-8)
            ),
            'candidate_log_energy_dtype': 'float64',
            'candidate_log_energy_shape': (
                'trajectory_by_step_by_action'
            ),
            'log_artifact_padding': 'NaN',
            'provenance': provenance,
            'scores': {
                'legacy_retrospective_compatibility': {
                    'type': 'retrospective_compatibility_score',
                    'family': 'unnormalized_diagnostic',
                    'artifact': 'step_log_score_test.npy',
                    'split': 'test',
                    'aggregation': ['trajectory_macro', 'valid_step_micro'],
                    'normalized': False,
                    'training_objective': True,
                },
                'retrospective_action_conditional': {
                    'type': (
                        'retrospective_action_conditional_log_probability'
                    ),
                    'family': 'action_conditional_log_probability',
                    'artifact': (
                        'step_log_score_retrospective_normalized_test.npy'
                    ),
                    'split': 'test',
                    'aggregation': ['trajectory_macro', 'valid_step_micro'],
                    'normalized': True,
                    'training_objective': False,
                },
            },
        })
    with open(os.path.join(run_dir, 'run_manifest.json'), 'w') as fout:
        json.dump(manifest, fout, indent=2)

    output_columns = [
        'num_trajs', 'fold', 'train_ll', 'test_ll', 'seed', 'fold_seed', 'gate_mode',
        'score_type', 'train_step_ll', 'test_step_ll', 'train_steps', 'test_steps',
        'iterations', 'stop_reason', 'final_loss', 'status',
    ]
    if normalized_mode:
        output_columns.extend([
            'normalized_score_type', 'normalized_test_traj_ll',
            'normalized_test_step_ll', 'normalized_test_steps',
            'log_normalizer_mean', 'fraction_z_gt_one',
            'candidate_gate_max_error', 'candidate_responsibility_max_error',
            'candidate_normalization_max_error',
        ])
    output_df = pd.DataFrame(columns=output_columns)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10015)
    completed_folds = []
    for fold_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs)):
        if args.fold_idx >= 0 and fold_idx != args.fold_idx:
            continue
        fold_seed = args.rand_seed + 1009 * fold_idx
        set_seed(fold_seed)
        train_trajs = [trajs[idx] for idx in train_idxes]
        test_trajs = [trajs[idx] for idx in test_idxes]
        model = PGIAVI(
            num_latents=args.num_latents,
            num_states=envr.num_states,
            num_actions=envr.num_actions,
            train_trajs=train_trajs,
            test_trajs=test_trajs,
            P=envr.P,
            discount=envr.gamma,
            gate_mode=args.gate_mode,
            loss_threshold=args.loss_threshold,
            max_iterations=args.max_iterations,
            retrospective_score_mode=args.retrospective_score_mode,
        )
        ll, outputs, agents = model.fit()

        param_dir = os.path.join(run_dir, f'{len(trajs)}/fold_{fold_idx}')
        os.makedirs(param_dir, exist_ok=True)
        with open(os.path.join(param_dir, 'train_idxes.json'), 'w') as fout:
            json.dump(train_idxes.tolist(), fout)
        with open(os.path.join(param_dir, 'test_idxes.json'), 'w') as fout:
            json.dump(test_idxes.tolist(), fout)

        gate_train, mask_train = pad_sequences(outputs['gate_train'])
        gate_test, mask_test = pad_sequences(outputs['gate_test'])
        responsibility_train, _ = pad_sequences(outputs['responsibility_train'])
        responsibility_test, _ = pad_sequences(outputs['responsibility_test'])
        step_score_train, _ = pad_sequences(
            outputs['step_log_score_train'], fill_value=np.nan
        )
        step_score_test, _ = pad_sequences(
            outputs['step_log_score_test'], fill_value=np.nan
        )
        latent_train, _ = pad_sequences(
            [latents[idx] for idx in train_idxes], fill_value=-1, dtype=np.int64
        )
        latent_test, _ = pad_sequences(
            [latents[idx] for idx in test_idxes], fill_value=-1, dtype=np.int64
        )

        np.save(os.path.join(param_dir, 'gate_train.npy'), gate_train)
        np.save(os.path.join(param_dir, 'gate_test.npy'), gate_test)
        np.save(os.path.join(param_dir, 'responsibility_train.npy'), responsibility_train)
        np.save(os.path.join(param_dir, 'responsibility_test.npy'), responsibility_test)
        np.save(os.path.join(param_dir, 'step_log_score_train.npy'), step_score_train)
        np.save(os.path.join(param_dir, 'step_log_score_test.npy'), step_score_test)
        np.save(os.path.join(param_dir, 'mask_train.npy'), mask_train)
        np.save(os.path.join(param_dir, 'mask_test.npy'), mask_test)
        np.save(os.path.join(param_dir, 'latent_train.npy'), latent_train)
        np.save(os.path.join(param_dir, 'latent_test.npy'), latent_test)
        if normalized_mode:
            normalized_step_score_test, _ = pad_sequences(
                outputs['step_log_score_retrospective_normalized_test'],
                fill_value=np.nan,
                dtype=np.float64,
            )
            retrospective_log_normalizer_test, _ = pad_sequences(
                outputs['retrospective_log_normalizer_test'],
                fill_value=np.nan,
                dtype=np.float64,
            )
            candidate_log_energy_test, _ = pad_sequences(
                outputs['candidate_log_energy_test'],
                fill_value=np.nan,
                dtype=np.float64,
            )
            candidate_observed_gate_test, _ = pad_sequences(
                outputs['candidate_observed_gate_test']
            )
            candidate_observed_responsibility_test, _ = pad_sequences(
                outputs['candidate_observed_responsibility_test']
            )
            np.save(
                os.path.join(
                    param_dir,
                    'step_log_score_retrospective_normalized_test.npy',
                ),
                normalized_step_score_test,
            )
            np.save(
                os.path.join(
                    param_dir, 'retrospective_log_normalizer_test.npy'
                ),
                retrospective_log_normalizer_test,
            )
            np.save(
                os.path.join(param_dir, 'candidate_log_energy_test.npy'),
                candidate_log_energy_test,
            )
            np.save(
                os.path.join(param_dir, 'candidate_observed_gate_test.npy'),
                candidate_observed_gate_test,
            )
            np.save(
                os.path.join(
                    param_dir,
                    'candidate_observed_responsibility_test.npy',
                ),
                candidate_observed_responsibility_test,
            )
            save_checkpoint(
                model, os.path.join(param_dir, 'intention_state_dict.pt')
            )
        for agent_idx, agent in enumerate(agents):
            np.save(os.path.join(param_dir, f'r_{agent_idx}.npy'), agent.r)
            np.save(os.path.join(param_dir, f'q_{agent_idx}.npy'), agent.q)

        output_row = [
            len(trajs), fold_idx, ll['train'], ll['test'], args.rand_seed,
            fold_seed, args.gate_mode, ll['score_type'], ll['train_step_mean'],
            ll['test_step_mean'], ll['train_steps'], ll['test_steps'],
            ll['iterations'], ll['stop_reason'], ll['final_loss'], 'complete',
        ]
        if normalized_mode:
            output_row.extend([
                ll['retrospective_normalized_score_type'],
                ll['test_retrospective_normalized_traj_mean'],
                ll['test_retrospective_normalized_step_mean'],
                ll['test_retrospective_normalized_steps'],
                ll['test_retrospective_log_normalizer_mean'],
                ll['test_retrospective_fraction_z_gt_one'],
                ll['candidate_gate_max_error'],
                ll['candidate_responsibility_max_error'],
                ll['candidate_normalization_max_error'],
            ])
            with open(os.path.join(param_dir, 'fold_manifest.json'), 'w') as fout:
                json.dump({
                    'fold': fold_idx,
                    'fold_seed': fold_seed,
                    'checkpoint_provenance': (
                        'recreated_submitted_configuration'
                    ),
                    'checkpoint': 'intention_state_dict.pt',
                    'legacy_policy_log_row_normalizer': (
                        ll['legacy_policy_log_row_normalizer']
                    ),
                    'candidate_gate_max_error': ll['candidate_gate_max_error'],
                    'candidate_responsibility_max_error': (
                        ll['candidate_responsibility_max_error']
                    ),
                    'candidate_normalization_max_error': (
                        ll['candidate_normalization_max_error']
                    ),
                }, fout, indent=2)
            write_sha256_manifest(param_dir)
        output_df.loc[len(output_df)] = output_row
        output_df.to_csv(os.path.join(run_dir, args.ll_filename), index=False)
        completed_folds.append(fold_idx)

    manifest['status'] = 'complete'
    manifest['completed_folds'] = completed_folds
    with open(os.path.join(run_dir, 'run_manifest.json'), 'w') as fout:
        json.dump(manifest, fout, indent=2)
