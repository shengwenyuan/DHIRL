"""Validate and summarize retrospective conditional-energy artifacts."""

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src_autotest.evaluate_sequence_p0 import (
    bootstrap_score_delta,
    empty_assignment_stats,
    load_experiment as load_p0_experiment,
    summarize_assignment,
    update_assignment_stats,
)


ATOL = 1e-6
LOG_REPLAY_ATOL = 1e-5
NULL_SAMPLES = 200
NULL_SEED = 20260726
EVENT_STATES = {'water_state_116': 116, 'home_state_0': 0}
EXPECTED_FORMAL_SEEDS = {
    'gridworld': {42},
    'labyrinth': {0, 42, 2026},
}


def parse_labeled_path(value):
    label, separator, path = value.partition('=')
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError('Expected LABEL=PATH.')
    return label, Path(path).expanduser()


def read_json(path):
    with path.open() as fin:
        return json.load(fin)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as fin:
        for block in iter(lambda: fin.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_sha256_manifest(fold_dir):
    manifest_path = fold_dir / 'SHA256SUMS'
    if not manifest_path.is_file():
        raise ValueError(f'{fold_dir}: missing SHA256SUMS.')
    checked = []
    for line in manifest_path.read_text().splitlines():
        digest, separator, filename = line.partition('  ')
        if not separator or len(digest) != 64 or not filename:
            raise ValueError(f'{manifest_path}: malformed line {line!r}.')
        path = fold_dir / filename
        if not path.is_file():
            raise ValueError(f'{manifest_path}: missing listed file {filename}.')
        if sha256_file(path) != digest:
            raise ValueError(f'{manifest_path}: hash mismatch for {filename}.')
        checked.append(filename)
    required = {
        'candidate_log_energy_test.npy',
        'candidate_observed_gate_test.npy',
        'candidate_observed_responsibility_test.npy',
        'fold_manifest.json',
        'gate_test.npy',
        'intention_state_dict.pt',
        'mask_test.npy',
        'responsibility_test.npy',
        'retrospective_log_normalizer_test.npy',
        'step_log_score_retrospective_normalized_test.npy',
        'step_log_score_test.npy',
        'test_idxes.json',
    }
    missing = required.difference(checked)
    if missing:
        raise ValueError(
            f'{manifest_path}: required files absent from checksum list: '
            f'{sorted(missing)}.'
        )
    present = {
        path.name
        for path in fold_dir.iterdir()
        if path.is_file() and path.name != 'SHA256SUMS'
    }
    if set(checked) != present:
        raise ValueError(
            f'{manifest_path}: checksum inventory differs from directory; '
            f'unlisted={sorted(present.difference(checked))}, '
            f'missing={sorted(set(checked).difference(present))}.'
        )
    return len(checked)


def validate_tree_sha256(root):
    manifest_path = root / 'SHA256SUMS'
    if not manifest_path.is_file():
        raise ValueError(f'{root}: missing root SHA256SUMS.')
    checked = set()
    for line in manifest_path.read_text().splitlines():
        digest, separator, relative = line.partition('  ')
        if not separator or len(digest) != 64 or not relative:
            raise ValueError(f'{manifest_path}: malformed line {line!r}.')
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'{manifest_path}: invalid entry {relative!r}.')
        checked.add(relative)
    present = {
        str(path.relative_to(root))
        for path in root.rglob('*')
        if path.is_file() and path != manifest_path
    }
    if checked != present:
        raise ValueError(
            f'{manifest_path}: recursive inventory mismatch; '
            f'unlisted={sorted(present - checked)[:10]}, '
            f'missing={sorted(checked - present)[:10]}.'
        )
    return len(checked)


def validate_formal_provenance(label, manifest, experiment_dir):
    provenance = manifest.get('provenance', {})
    required = {
        'purpose': 'formal_rebuttal',
        'git_branch': '26r',
        'git_dirty': False,
        'baseline_ancestor': True,
    }
    for key, expected in required.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f'{label}: provenance {key}={provenance.get(key)!r}, '
                f'expected {expected!r}.'
            )
    for key in (
        'git_commit', 'baseline_tag', 'config_path', 'config_sha256',
        'runner_log_path', 'runner_log_sha256', 'runner_manifest_path',
        'runner_manifest_sha256', 'python_executable', 'command',
    ):
        if not provenance.get(key):
            raise ValueError(f'{label}: missing provenance field {key}.')

    file_bindings = (
        ('config_path', 'config_sha256'),
        ('runner_log_path', 'runner_log_sha256'),
        ('runner_manifest_path', 'runner_manifest_sha256'),
    )
    for path_key, hash_key in file_bindings:
        path = Path(provenance[path_key])
        if not path.is_file() or sha256_file(path) != provenance[hash_key]:
            raise ValueError(f'{label}: invalid provenance binding {path_key}.')

    runner_manifest = read_json(Path(provenance['runner_manifest_path']))
    for key in ('git_commit', 'git_branch', 'git_dirty', 'baseline_tag'):
        if runner_manifest.get(key) != provenance.get(key):
            raise ValueError(
                f'{label}: runner/trainer provenance differs for {key}.'
            )
    if (
        runner_manifest.get('config_sha256') != provenance['config_sha256']
        or runner_manifest.get('status') != 'complete'
    ):
        raise ValueError(f'{label}: runner manifest is not a matching success.')
    if subprocess.run(
        [
            'git', 'merge-base', '--is-ancestor',
            provenance['baseline_tag'], provenance['git_commit'],
        ],
        check=False,
    ).returncode != 0:
        raise ValueError(f'{label}: baseline ancestry cannot be reproduced.')
    return validate_tree_sha256(experiment_dir)


def validate_mask(mask, fold_dir):
    if mask.ndim != 2:
        raise ValueError(f'{fold_dir}: mask_test.npy must be two-dimensional.')
    if mask.dtype != np.bool_:
        if not np.isin(mask, (0, 1)).all():
            raise ValueError(f'{fold_dir}: mask_test.npy is not boolean.')
        mask = mask.astype(bool)
    lengths = mask.sum(axis=1)
    if (lengths <= 0).any():
        raise ValueError(f'{fold_dir}: every trajectory needs a valid step.')
    expected = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    if not np.array_equal(mask, expected):
        raise ValueError(f'{fold_dir}: masks must be contiguous prefixes.')
    return mask, lengths


def validate_probability_artifact(values, mask, num_latents, name, fold_dir):
    if values.shape != (*mask.shape, num_latents):
        raise ValueError(f'{fold_dir}: invalid {name} shape {values.shape}.')
    valid = values[mask]
    if not np.isfinite(valid).all() or (valid < -ATOL).any():
        raise ValueError(f'{fold_dir}: invalid probability in {name}.')
    if not np.allclose(valid.sum(axis=-1), 1.0, atol=ATOL, rtol=0):
        raise ValueError(f'{fold_dir}: {name} is not normalized.')
    if not np.allclose(values[~mask], 0.0, atol=ATOL, rtol=0):
        raise ValueError(f'{fold_dir}: {name} padding must be zero.')


def validate_log_artifact(values, mask, name, fold_dir, trailing_shape=()):
    expected_shape = (*mask.shape, *trailing_shape)
    if values.shape != expected_shape:
        raise ValueError(
            f'{fold_dir}: {name} has shape {values.shape}, expected {expected_shape}.'
        )
    if not np.isfinite(values[mask]).all():
        raise ValueError(f'{fold_dir}: {name} has non-finite valid values.')
    if not np.isnan(values[~mask]).all():
        raise ValueError(f'{fold_dir}: {name} padding must be NaN.')


def event_positions(states, state, unit):
    positions = np.flatnonzero(states == state)
    if unit == 'state_occupancy_timestep':
        return positions
    if unit == 'visit_entry':
        return positions[
            (positions == 0) | (states[np.maximum(positions - 1, 0)] != state)
        ]
    raise ValueError(f'Unknown event unit {unit!r}.')


def boundary_distances(labels, positions):
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    if not len(positions):
        return np.empty(0, dtype=np.int64), 0
    if not len(boundaries):
        return np.empty(0, dtype=np.int64), len(positions)
    distances = np.min(
        np.abs(positions[:, None] - boundaries[None, :]), axis=1
    )
    return distances, 0


def boundary_coverage(labels, positions, tolerance):
    distances, no_boundary_events = boundary_distances(labels, positions)
    return (
        int(np.sum(distances <= tolerance)),
        int(len(distances) + no_boundary_events),
    )


def event_summary(records, trajectories, num_latents):
    units = ('state_occupancy_timestep', 'visit_entry')
    observed = {
        unit: {
            name: {tolerance: [0, 0] for tolerance in (0, 5, 10)}
            for name in EVENT_STATES
        }
        for unit in units
    }
    observed_distances = {
        unit: {
            name: {'finite': [], 'no_boundary_events': 0}
            for name in EVENT_STATES
        }
        for unit in units
    }
    fold_counts = {}
    for trajectory_id, record in enumerate(records):
        labels = record['gate']
        states = np.asarray(
            [step[0] for step in trajectories[trajectory_id]], dtype=np.int64
        )
        fold = record['fold']
        counts = fold_counts.setdefault(
            fold,
            {
                'steps': np.zeros(num_latents, dtype=np.int64),
                'events': {
                    name: np.zeros(num_latents, dtype=np.int64)
                    for name in EVENT_STATES
                },
            },
        )
        counts['steps'] += np.bincount(labels, minlength=num_latents)
        for name, state in EVENT_STATES.items():
            positions = event_positions(
                states, state, 'state_occupancy_timestep'
            )
            counts['events'][name] += np.bincount(
                labels[positions], minlength=num_latents
            )
            for unit in units:
                unit_positions = event_positions(states, state, unit)
                distances, no_boundary_events = boundary_distances(
                    labels, unit_positions
                )
                observed_distances[unit][name]['finite'].append(distances)
                observed_distances[unit][name][
                    'no_boundary_events'
                ] += no_boundary_events
                for tolerance in (0, 5, 10):
                    matched = int(np.sum(distances <= tolerance))
                    total = int(len(distances) + no_boundary_events)
                    observed[unit][name][tolerance][0] += matched
                    observed[unit][name][tolerance][1] += total

    enrichment = {}
    for name in EVENT_STATES:
        by_fold = []
        for fold, counts in sorted(fold_counts.items()):
            event_counts = counts['events'][name]
            total_events = int(event_counts.sum())
            total_steps = int(counts['steps'].sum())
            baseline = total_events / total_steps if total_steps else 0.0
            rates = np.divide(
                event_counts,
                counts['steps'],
                out=np.zeros(num_latents, dtype=float),
                where=counts['steps'] > 0,
            )
            ratios = rates / baseline if baseline else np.zeros_like(rates)
            by_fold.append({
                'fold': fold,
                'events': total_events,
                'component_event_counts': event_counts.tolist(),
                'component_step_counts': counts['steps'].tolist(),
                'component_enrichment_ratio': ratios.tolist(),
                'max_enrichment_ratio': float(ratios.max(initial=0.0)),
            })
        enrichment[name] = by_fold

    rng = np.random.default_rng(NULL_SEED)
    null = {
        unit: {
            name: {tolerance: [] for tolerance in (0, 5, 10)}
            for name in EVENT_STATES
        }
        for unit in units
    }
    for _ in range(NULL_SAMPLES):
        matched_counts = {
            unit: {
                name: {tolerance: 0 for tolerance in (0, 5, 10)}
                for name in EVENT_STATES
            }
            for unit in units
        }
        total_counts = {
            unit: {name: 0 for name in EVENT_STATES}
            for unit in units
        }
        for trajectory_id, record in enumerate(records):
            labels = record['gate']
            shift = int(rng.integers(0, len(labels)))
            shifted = np.roll(labels, shift)
            states = np.asarray(
                [step[0] for step in trajectories[trajectory_id]],
                dtype=np.int64,
            )
            for name, state in EVENT_STATES.items():
                for unit in units:
                    positions = event_positions(states, state, unit)
                    total_counts[unit][name] += len(positions)
                    for tolerance in (0, 5, 10):
                        matched, _ = boundary_coverage(
                            shifted, positions, tolerance
                        )
                        matched_counts[unit][name][tolerance] += matched
        for unit in units:
            for name in EVENT_STATES:
                for tolerance in (0, 5, 10):
                    denominator = total_counts[unit][name]
                    null[unit][name][tolerance].append(
                        matched_counts[unit][name][tolerance] / denominator
                        if denominator
                        else 0.0
                    )

    coverage = {}
    nearest_distance = {}
    for unit in units:
        coverage[unit] = {}
        nearest_distance[unit] = {}
        for name in EVENT_STATES:
            coverage[unit][name] = {}
            for tolerance in (0, 5, 10):
                matched, total = observed[unit][name][tolerance]
                observed_rate = matched / total if total else 0.0
                null_values = np.asarray(
                    null[unit][name][tolerance], dtype=float
                )
                coverage[unit][name][str(tolerance)] = {
                    'matched': matched,
                    'events': total,
                    'observed_rate': observed_rate,
                    'circular_shift_null_mean': float(null_values.mean()),
                    'circular_shift_null_95_interval': np.quantile(
                        null_values, [0.025, 0.975]
                    ).tolist(),
                    'one_sided_p': float(
                        (1 + np.sum(null_values >= observed_rate))
                        / (NULL_SAMPLES + 1)
                    ),
                }
            finite_parts = observed_distances[unit][name]['finite']
            finite = (
                np.concatenate(finite_parts)
                if finite_parts
                else np.empty(0, dtype=np.int64)
            )
            no_boundary_events = observed_distances[unit][name][
                'no_boundary_events'
            ]
            nearest_distance[unit][name] = {
                'events': int(len(finite) + no_boundary_events),
                'finite_distance_events': int(len(finite)),
                'no_boundary_events': int(no_boundary_events),
                'mean': float(finite.mean()) if len(finite) else None,
                'std': float(finite.std()) if len(finite) else None,
                'quantiles': (
                    {
                        name: float(value)
                        for name, value in zip(
                            ('p00', 'p25', 'p50', 'p75', 'p95', 'p100'),
                            np.quantile(
                                finite, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]
                            ),
                        )
                    }
                    if len(finite)
                    else None
                ),
            }
    return {
        'event_definition': 'current state equals the named canonical node',
        'event_units': {
            'state_occupancy_timestep': (
                'every timestep occupying the event state'
            ),
            'visit_entry': (
                'first timestep of each contiguous visit to the event state'
            ),
        },
        'assignment': 'retrospective_gate_argmax',
        'component_enrichment_event_unit': 'state_occupancy_timestep',
        'component_enrichment_by_fold': enrichment,
        'boundary_event_coverage': coverage,
        'nearest_boundary_distance': nearest_distance,
        'null_samples': NULL_SAMPLES,
        'null_seed': NULL_SEED,
    }


def load_energy_experiment(label, experiment_dir):
    experiment_dir = experiment_dir.resolve()
    manifest = read_json(experiment_dir / 'run_manifest.json')
    required_manifest = {
        'artifact_schema': 'p0_sequence_artifacts_v2',
        'checkpoint_provenance': 'recreated_submitted_configuration',
        'training_objective': 'legacy_retrospective_compatibility',
        'retrospective_score_mode': 'action_normalized',
        'candidate_context_rule': (
            'observed_prefix_fixed_current_action_enumerated'
        ),
        'candidate_log_energy_dtype': 'float64',
        'log_artifact_padding': 'NaN',
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f'{label}: manifest {key}={manifest.get(key)!r}, '
                f'expected {expected!r}.'
            )
    if manifest.get('status') != 'complete':
        raise ValueError(f'{label}: run is not complete.')
    if manifest.get('gate_mode') != 'retrospective':
        raise ValueError(f'{label}: gate_mode must be retrospective.')
    if manifest.get('model_type') != 'IntentionRNN':
        raise ValueError(f'{label}: model_type must be IntentionRNN.')
    if manifest.get('num_repeats') != 1:
        raise ValueError(f'{label}: num_repeats must be one.')
    root_checked_hash_files = 0
    if int(manifest['fold_idx']) < 0:
        root_checked_hash_files = validate_formal_provenance(
            label, manifest, experiment_dir
        )

    trajectory_path = Path(manifest['trajectory_path'])
    if sha256_file(trajectory_path) != manifest['trajectory_sha256']:
        raise ValueError(f'{label}: trajectory input hash mismatch.')
    trajectories = read_json(trajectory_path)
    trajectories = trajectories[:int(manifest['num_trajs_used'])]
    num_trajs = len(trajectories)
    num_actions = int(manifest['num_actions'])
    num_latents = int(manifest['num_latents'])
    legacy_scale = float(manifest['legacy_policy_log_scale'])

    fold_dirs = sorted(
        (experiment_dir / str(num_trajs)).glob('fold_*'),
        key=lambda path: int(path.name.removeprefix('fold_')),
    )
    expected_folds = (
        [int(manifest['fold_idx'])]
        if int(manifest['fold_idx']) >= 0
        else list(range(int(manifest['num_folds'])))
    )
    observed_folds = [
        int(path.name.removeprefix('fold_')) for path in fold_dirs
    ]
    if observed_folds != expected_folds:
        raise ValueError(
            f'{label}: folds {observed_folds}, expected {expected_folds}.'
        )

    records = [None] * num_trajs
    observed_ids = set()
    gate_stats = empty_assignment_stats(num_latents)
    responsibility_stats = empty_assignment_stats(num_latents)
    max_errors = {
        'runtime_candidate_gate_parity': 0.0,
        'runtime_responsibility_parity': 0.0,
        'runtime_normalization': 0.0,
        'saved_candidate_gate_parity': 0.0,
        'saved_candidate_responsibility_parity': 0.0,
        'candidate_observed_vs_legacy_minus_scale': 0.0,
        'normalized_score_identity': 0.0,
        'action_probability_mass': 0.0,
    }
    checked_hash_files = 0
    for fold_dir in fold_dirs:
        checked_hash_files += validate_sha256_manifest(fold_dir)
        fold_manifest = read_json(fold_dir / 'fold_manifest.json')
        runtime_errors = {
            'runtime_candidate_gate_parity': float(
                fold_manifest['candidate_gate_max_error']
            ),
            'runtime_responsibility_parity': float(
                fold_manifest['candidate_responsibility_max_error']
            ),
            'runtime_normalization': float(
                fold_manifest['candidate_normalization_max_error']
            ),
        }
        for name in (
            'runtime_candidate_gate_parity',
            'runtime_responsibility_parity',
            'runtime_normalization',
        ):
            if runtime_errors[name] >= ATOL:
                raise ValueError(f'{fold_dir}: {name} exceeds {ATOL}.')
        for name, value in runtime_errors.items():
            max_errors[name] = max(max_errors[name], value)
        fold_idx = int(fold_dir.name.removeprefix('fold_'))
        test_idxes = np.asarray(
            read_json(fold_dir / 'test_idxes.json'), dtype=np.int64
        )
        if len(test_idxes) != len(np.unique(test_idxes)):
            raise ValueError(f'{fold_dir}: duplicate test trajectory IDs.')
        duplicates = observed_ids.intersection(test_idxes.tolist())
        if duplicates:
            raise ValueError(f'{label}: duplicate OOF IDs {sorted(duplicates)[:5]}.')

        mask, lengths = validate_mask(np.load(fold_dir / 'mask_test.npy'), fold_dir)
        if len(test_idxes) != mask.shape[0]:
            raise ValueError(f'{fold_dir}: index and array counts differ.')
        gate = np.load(fold_dir / 'gate_test.npy')
        responsibility = np.load(fold_dir / 'responsibility_test.npy')
        candidate_gate = np.load(
            fold_dir / 'candidate_observed_gate_test.npy'
        )
        candidate_responsibility = np.load(
            fold_dir / 'candidate_observed_responsibility_test.npy'
        )
        legacy = np.load(fold_dir / 'step_log_score_test.npy')
        normalized = np.load(
            fold_dir / 'step_log_score_retrospective_normalized_test.npy'
        )
        log_normalizer = np.load(
            fold_dir / 'retrospective_log_normalizer_test.npy'
        )
        candidate_energy = np.load(
            fold_dir / 'candidate_log_energy_test.npy'
        )
        validate_probability_artifact(
            gate, mask, num_latents, 'gate_test.npy', fold_dir
        )
        validate_probability_artifact(
            responsibility,
            mask,
            num_latents,
            'responsibility_test.npy',
            fold_dir,
        )
        validate_probability_artifact(
            candidate_gate,
            mask,
            num_latents,
            'candidate_observed_gate_test.npy',
            fold_dir,
        )
        validate_probability_artifact(
            candidate_responsibility,
            mask,
            num_latents,
            'candidate_observed_responsibility_test.npy',
            fold_dir,
        )
        saved_gate_error = float(
            np.max(np.abs(candidate_gate[mask] - gate[mask]))
        )
        saved_responsibility_error = float(
            np.max(
                np.abs(
                    candidate_responsibility[mask] - responsibility[mask]
                )
            )
        )
        max_errors['saved_candidate_gate_parity'] = max(
            max_errors['saved_candidate_gate_parity'], saved_gate_error
        )
        max_errors['saved_candidate_responsibility_parity'] = max(
            max_errors['saved_candidate_responsibility_parity'],
            saved_responsibility_error,
        )
        if (
            saved_gate_error >= ATOL
            or saved_responsibility_error >= ATOL
        ):
            raise ValueError(
                f'{fold_dir}: saved candidate assignment parity failed.'
            )
        if not np.array_equal(
            np.argmax(candidate_gate[mask], axis=-1),
            np.argmax(gate[mask], axis=-1),
        ):
            raise ValueError(
                f'{fold_dir}: candidate/reference hard gate differs.'
            )
        if not np.array_equal(
            np.argmax(candidate_responsibility[mask], axis=-1),
            np.argmax(responsibility[mask], axis=-1),
        ):
            raise ValueError(
                f'{fold_dir}: candidate/reference hard responsibility differs.'
            )
        validate_log_artifact(legacy, mask, 'step_log_score_test.npy', fold_dir)
        validate_log_artifact(
            normalized,
            mask,
            'step_log_score_retrospective_normalized_test.npy',
            fold_dir,
        )
        validate_log_artifact(
            log_normalizer,
            mask,
            'retrospective_log_normalizer_test.npy',
            fold_dir,
        )
        validate_log_artifact(
            candidate_energy,
            mask,
            'candidate_log_energy_test.npy',
            fold_dir,
            trailing_shape=(num_actions,),
        )
        if candidate_energy.dtype != np.float64:
            raise ValueError(f'{fold_dir}: candidate energy must be float64.')

        valid_energy = candidate_energy[mask]
        valid_logz = log_normalizer[mask]
        mass_error = np.max(
            np.abs(np.exp(valid_energy - valid_logz[:, None]).sum(axis=-1) - 1)
        )
        max_errors['action_probability_mass'] = max(
            max_errors['action_probability_mass'], float(mass_error)
        )
        if mass_error >= ATOL:
            raise ValueError(f'{fold_dir}: action probabilities do not normalize.')
        if np.max(normalized[mask]) > ATOL:
            raise ValueError(f'{fold_dir}: normalized log score exceeds zero.')

        for row, trajectory_id in enumerate(test_idxes):
            trajectory_id = int(trajectory_id)
            length = int(lengths[row])
            trajectory = trajectories[trajectory_id]
            if len(trajectory) != length:
                raise ValueError(
                    f'{fold_dir}: trajectory {trajectory_id} length mismatch.'
                )
            actions = np.asarray(
                [step[1] for step in trajectory], dtype=np.int64
            )
            if ((actions < 0) | (actions >= num_actions)).any():
                raise ValueError(f'{fold_dir}: action outside support.')
            observed_energy = candidate_energy[
                row, np.arange(length), actions
            ]
            legacy_values = legacy[row, :length].astype(np.float64)
            normalized_values = normalized[row, :length].astype(np.float64)
            logz_values = log_normalizer[row, :length].astype(np.float64)
            replay_error = np.max(
                np.abs(observed_energy - (legacy_values - legacy_scale))
            )
            identity_error = np.max(
                np.abs(normalized_values - (observed_energy - logz_values))
            )
            max_errors['candidate_observed_vs_legacy_minus_scale'] = max(
                max_errors['candidate_observed_vs_legacy_minus_scale'],
                float(replay_error),
            )
            max_errors['normalized_score_identity'] = max(
                max_errors['normalized_score_identity'], float(identity_error)
            )
            if replay_error >= LOG_REPLAY_ATOL or identity_error >= ATOL:
                raise ValueError(
                    f'{fold_dir}: scorer identity failed for trajectory '
                    f'{trajectory_id}.'
                )
            gate_labels = update_assignment_stats(
                gate_stats, gate[row, :length]
            )
            responsibility_labels = update_assignment_stats(
                responsibility_stats, responsibility[row, :length]
            )
            records[trajectory_id] = {
                'fold': fold_idx,
                'gate': gate_labels,
                'responsibility': responsibility_labels,
                'step_score': normalized_values,
                'legacy_step_score': legacy_values,
                'log_normalizer': logz_values,
            }
        observed_ids.update(test_idxes.tolist())

    expected_ids = set(range(num_trajs))
    if int(manifest['fold_idx']) < 0 and observed_ids != expected_ids:
        raise ValueError(f'{label}: OOF coverage is not exact.')
    if int(manifest['fold_idx']) >= 0:
        records = [record for record in records if record is not None]
    normalized_trajectory_means = np.asarray([
        record['step_score'].mean() for record in records
    ])
    legacy_trajectory_means = np.asarray([
        record['legacy_step_score'].mean() for record in records
    ])
    normalized_values = np.concatenate([
        record['step_score'] for record in records
    ])
    legacy_values = np.concatenate([
        record['legacy_step_score'] for record in records
    ])
    logz_values = np.concatenate([
        record['log_normalizer'] for record in records
    ])
    summary = {
        'path': str(experiment_dir),
        'domain': manifest['domain'],
        'seed': int(manifest['rand_seed']),
        'manifest': manifest,
        'validation': {
            'exact_oof_coverage': int(manifest['fold_idx']) < 0,
            'folds': observed_folds,
            'num_trajectories': len(records),
            'num_valid_steps': int(normalized_values.size),
            'full_action_support': list(range(num_actions)),
            'candidate_policy_normalized': True,
            'padding_contract': True,
            'checkpoint_provenance': manifest['checkpoint_provenance'],
            'checked_sha256_files': checked_hash_files,
            'checked_root_sha256_files': root_checked_hash_files,
            'same_run_assignment_parity': {
                'hard_gate_agreement': 1.0,
                'hard_responsibility_agreement': 1.0,
                'gate_ari': 1.0,
                'responsibility_ari': 1.0,
                'transition_duration_occupancy_equal': True,
            },
            'max_errors': max_errors,
        },
        'scores': {
            'legacy_compatibility': {
                'trajectory_macro_mean': float(legacy_trajectory_means.mean()),
                'valid_step_micro_mean': float(legacy_values.mean()),
            },
            'retrospective_action_conditional': {
                'type': (
                    'retrospective_action_conditional_log_probability'
                ),
                'family': 'action_conditional_log_probability',
                'trajectory_macro_mean': float(
                    normalized_trajectory_means.mean()
                ),
                'valid_step_micro_mean': float(normalized_values.mean()),
                'uniform_baseline': float(-np.log(num_actions)),
            },
            'log_normalizer': {
                'mean': float(logz_values.mean()),
                'std': float(logz_values.std()),
                'min': float(logz_values.min()),
                'quantiles': {
                    name: float(value)
                    for name, value in zip(
                        ('p05', 'p25', 'p50', 'p75', 'p95'),
                        np.quantile(
                            logz_values, [0.05, 0.25, 0.5, 0.75, 0.95]
                        ),
                    )
                },
                'max': float(logz_values.max()),
                'fraction_z_gt_one': float(np.mean(logz_values > 0)),
            },
        },
        'gate': summarize_assignment(gate_stats),
        'responsibility': summarize_assignment(responsibility_stats),
    }
    if manifest['domain'] == 'labyrinth' and int(manifest['fold_idx']) < 0:
        summary['events'] = event_summary(records, trajectories, num_latents)
    return {
        'manifest': manifest,
        'records': records,
        'summary': summary,
    }


def comparison_manifest_differences(left_manifest, right_manifest):
    keys = [
        'domain', 'rand_seed', 'num_trajs_used', 'num_latents',
        'num_actions', 'num_folds', 'fold_random_state', 'fold_seeds',
        'trajectory_bytes', 'trajectory_sha256',
    ]
    if left_manifest.get('domain') == 'gridworld':
        keys.extend(['latent_bytes', 'latent_sha256'])
    if left_manifest.get('domain') == 'labyrinth':
        keys.extend(['transition_bytes', 'transition_sha256'])
    return [
        key for key in keys
        if (
            key not in left_manifest
            or key not in right_manifest
            or left_manifest[key] != right_manifest[key]
        )
    ]


def validate_record_alignment(
    left_label, left, right_label, right, allow_different_seed=False
):
    differences = comparison_manifest_differences(
        left['manifest'], right['manifest']
    )
    if allow_different_seed:
        differences = [
            key for key in differences if key not in {'rand_seed', 'fold_seeds'}
        ]
    if differences:
        raise ValueError(
            f'{left_label}/{right_label}: manifest fields differ: '
            f'{differences}.'
        )
    if len(left['records']) != len(right['records']):
        raise ValueError(f'{left_label}/{right_label}: trajectory count mismatch.')
    for trajectory_id, (left_record, right_record) in enumerate(
        zip(left['records'], right['records'])
    ):
        if left_record['fold'] != right_record['fold']:
            raise ValueError(
                f'{left_label}/{right_label}: fold mismatch for trajectory '
                f'{trajectory_id}.'
            )
        if len(left_record['step_score']) != len(right_record['step_score']):
            raise ValueError(
                f'{left_label}/{right_label}: length mismatch for trajectory '
                f'{trajectory_id}.'
            )


def pairwise_stability(left_label, left, right_label, right):
    if left['manifest']['domain'] != right['manifest']['domain']:
        return None
    validate_record_alignment(
        left_label,
        left,
        right_label,
        right,
        allow_different_seed=True,
    )
    folds = sorted({record['fold'] for record in left['records']})
    by_fold = []
    for fold in folds:
        pairs = [
            (left_record, right_record)
            for left_record, right_record in zip(
                left['records'], right['records']
            )
            if left_record['fold'] == fold
        ]
        if not pairs:
            continue
        left_gate = np.concatenate([pair[0]['gate'] for pair in pairs])
        right_gate = np.concatenate([pair[1]['gate'] for pair in pairs])
        left_responsibility = np.concatenate([
            pair[0]['responsibility'] for pair in pairs
        ])
        right_responsibility = np.concatenate([
            pair[1]['responsibility'] for pair in pairs
        ])
        by_fold.append({
            'fold': fold,
            'steps': int(left_gate.size),
            'gate_ari': float(adjusted_rand_score(left_gate, right_gate)),
            'gate_nmi': float(
                normalized_mutual_info_score(left_gate, right_gate)
            ),
            'responsibility_ari': float(
                adjusted_rand_score(
                    left_responsibility, right_responsibility
                )
            ),
            'responsibility_nmi': float(
                normalized_mutual_info_score(
                    left_responsibility, right_responsibility
                )
            ),
        })
    weights = np.asarray([item['steps'] for item in by_fold])
    return {
        'left': left_label,
        'right': right_label,
        'by_fold': by_fold,
        'gate_ari_step_weighted_mean': float(np.average(
            [item['gate_ari'] for item in by_fold], weights=weights
        )),
        'responsibility_ari_step_weighted_mean': float(np.average(
            [item['responsibility_ari'] for item in by_fold], weights=weights
        )),
        'gate_nmi_step_weighted_mean': float(np.average(
            [item['gate_nmi'] for item in by_fold], weights=weights
        )),
        'responsibility_nmi_step_weighted_mean': float(np.average(
            [item['responsibility_nmi'] for item in by_fold], weights=weights
        )),
    }


def compare_with_predictive_baseline(
    label, energy, baseline_path, expected_gate_mode
):
    baseline = load_p0_experiment(
        f'{label}_{expected_gate_mode}', baseline_path
    )
    manifest = baseline['manifest']
    if (
        manifest.get('score_type') != 'predictive_action_log_likelihood'
        or manifest.get('gate_mode') != expected_gate_mode
        or manifest.get('status') != 'complete'
        or int(manifest.get('fold_idx', 0)) != -1
        or manifest.get('completed_folds') != list(
            range(int(manifest.get('num_folds', 0)))
        )
    ):
        raise ValueError(
            f'{label}: comparison path is not a complete '
            f'{expected_gate_mode} predictive run.'
        )
    validate_record_alignment(
        f'{label}_{expected_gate_mode}', baseline, label, energy
    )
    result = bootstrap_score_delta(baseline['records'], energy['records'])
    result.update({
        'comparison': (
            'normalized_retrospective_minus_' + expected_gate_mode
        ),
        'baseline_gate_mode': expected_gate_mode,
        'baseline_score_type': 'predictive_action_log_likelihood',
        'baseline_trajectory_macro_mean': baseline['summary'][
            'step_score'
        ]['trajectory_macro_mean'],
        'baseline_valid_step_micro_mean': baseline['summary'][
            'step_score'
        ]['valid_step_micro_mean'],
        'right_score_type': (
            'retrospective_action_conditional_log_probability'
        ),
        'interpretation': (
            'descriptive_held_out_score_comparison_not_information_set_ablation'
        ),
    })
    return result


def assignment_summary_for_records(records, field, num_latents):
    stats = empty_assignment_stats(num_latents)
    for record in records:
        probabilities = np.eye(num_latents, dtype=np.float64)[record[field]]
        update_assignment_stats(stats, probabilities)
    return summarize_assignment(stats)


def assignment_shape_delta(historical_records, recreated_records, field, num_latents):
    historical = assignment_summary_for_records(
        historical_records, field, num_latents
    )
    recreated = assignment_summary_for_records(
        recreated_records, field, num_latents
    )
    historical_occupancy = np.asarray(historical['occupancy_sorted'])
    recreated_occupancy = np.asarray(recreated['occupancy_sorted'])
    return {
        'transition_rate_recreated_minus_historical': (
            recreated['transition_rate'] - historical['transition_rate']
        ),
        'segment_duration_mean_recreated_minus_historical': (
            recreated['segment_duration']['mean']
            - historical['segment_duration']['mean']
        ),
        'segment_duration_median_recreated_minus_historical': (
            recreated['segment_duration']['median']
            - historical['segment_duration']['median']
        ),
        'sorted_occupancy_l1_distance': float(
            np.abs(recreated_occupancy - historical_occupancy).sum()
        ),
        'sorted_occupancy_max_abs_distance': float(
            np.abs(recreated_occupancy - historical_occupancy).max()
        ),
        'historical': historical,
        'recreated': recreated,
    }


def fold_train_alignment(
    historical_path, recreated_path, num_trajs, fold, field, num_latents
):
    filename = f'{field}_train.npy'
    historical = np.load(
        historical_path / str(num_trajs) / f'fold_{fold}' / filename
    )
    recreated = np.load(
        recreated_path / str(num_trajs) / f'fold_{fold}' / filename
    )
    historical_mask = np.load(
        historical_path / str(num_trajs) / f'fold_{fold}' / 'mask_train.npy'
    ).astype(bool)
    recreated_mask = np.load(
        recreated_path / str(num_trajs) / f'fold_{fold}' / 'mask_train.npy'
    ).astype(bool)
    if (
        historical.shape != recreated.shape
        or not np.array_equal(historical_mask, recreated_mask)
    ):
        raise ValueError(
            f'Historical/recreated train arrays differ for fold {fold} {field}.'
        )
    historical_labels = np.argmax(historical[historical_mask], axis=-1)
    recreated_labels = np.argmax(recreated[recreated_mask], axis=-1)
    contingency = np.zeros(
        (num_latents, num_latents), dtype=np.int64
    )
    np.add.at(contingency, (historical_labels, recreated_labels), 1)
    historical_ids, recreated_ids = linear_sum_assignment(-contingency)
    recreated_to_historical = {
        int(recreated): int(historical)
        for historical, recreated in zip(historical_ids, recreated_ids)
    }
    return {
        'alignment_source': 'train_fold_valid_steps',
        'contingency_historical_by_recreated': contingency.tolist(),
        'recreated_to_historical': recreated_to_historical,
        'train_aligned_agreement': float(
            contingency[historical_ids, recreated_ids].sum()
            / contingency.sum()
        ),
    }


def compare_with_historical(label, energy, historical_path):
    historical_path = historical_path.resolve()
    historical = load_p0_experiment(
        f'{label}_historical_retrospective', historical_path
    )
    manifest = historical['manifest']
    if (
        manifest.get('gate_mode') != 'retrospective'
        or manifest.get('score_type') != 'retrospective_compatibility_score'
        or manifest.get('status') != 'complete'
        or int(manifest.get('fold_idx', 0)) != -1
    ):
        raise ValueError(
            f'{label}: historical path is not a complete retrospective run.'
        )
    validate_record_alignment(
        f'{label}_historical', historical, label, energy
    )
    num_latents = int(manifest['num_latents'])
    num_trajs = int(manifest['num_trajs_used'])
    folds = sorted({record['fold'] for record in energy['records']})
    by_fold = []
    for fold in folds:
        pairs = [
            (historical_record, recreated_record)
            for historical_record, recreated_record in zip(
                historical['records'], energy['records']
            )
            if historical_record['fold'] == fold
        ]
        row = {'fold': fold}
        for field in ('gate', 'responsibility'):
            historical_labels = np.concatenate([
                pair[0][field] for pair in pairs
            ])
            recreated_labels = np.concatenate([
                pair[1][field] for pair in pairs
            ])
            alignment = fold_train_alignment(
                historical_path,
                Path(energy['summary']['path']),
                num_trajs,
                fold,
                field,
                num_latents,
            )
            mapped = np.asarray([
                alignment['recreated_to_historical'][int(value)]
                for value in recreated_labels
            ])
            row[field] = {
                'steps': int(historical_labels.size),
                'ari': float(adjusted_rand_score(
                    historical_labels, recreated_labels
                )),
                'nmi': float(normalized_mutual_info_score(
                    historical_labels, recreated_labels
                )),
                'test_agreement_after_train_fold_alignment': float(
                    np.mean(mapped == historical_labels)
                ),
                'train_fold_alignment': alignment,
            }
        by_fold.append(row)
    weights = np.asarray([row['gate']['steps'] for row in by_fold])
    return {
        'historical_path': str(historical_path),
        'recreated_path': energy['summary']['path'],
        'score_comparison': (
            'prohibited_different_checkpoints_and_score_interpretations'
        ),
        'by_fold': by_fold,
        'gate_ari_step_weighted_mean': float(np.average(
            [row['gate']['ari'] for row in by_fold], weights=weights
        )),
        'gate_nmi_step_weighted_mean': float(np.average(
            [row['gate']['nmi'] for row in by_fold], weights=weights
        )),
        'gate_aligned_agreement_step_weighted_mean': float(np.average(
            [
                row['gate']['test_agreement_after_train_fold_alignment']
                for row in by_fold
            ],
            weights=weights,
        )),
        'responsibility_ari_step_weighted_mean': float(np.average(
            [row['responsibility']['ari'] for row in by_fold], weights=weights
        )),
        'responsibility_nmi_step_weighted_mean': float(np.average(
            [row['responsibility']['nmi'] for row in by_fold], weights=weights
        )),
        'responsibility_aligned_agreement_step_weighted_mean': float(np.average(
            [
                row['responsibility'][
                    'test_agreement_after_train_fold_alignment'
                ]
                for row in by_fold
            ],
            weights=weights,
        )),
        'gate_shape_delta': assignment_shape_delta(
            historical['records'], energy['records'], 'gate', num_latents
        ),
        'responsibility_shape_delta': assignment_shape_delta(
            historical['records'],
            energy['records'],
            'responsibility',
            num_latents,
        ),
    }


def seed_aggregate(experiments, allow_partial):
    by_domain = {}
    for label, experiment in experiments.items():
        by_domain.setdefault(experiment['manifest']['domain'], []).append(
            (label, experiment)
        )
    aggregate = {}
    for domain, items in by_domain.items():
        seeds = [int(item['manifest']['rand_seed']) for _, item in items]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f'{domain}: duplicate experiment seeds.')
        if not allow_partial:
            if set(seeds) != EXPECTED_FORMAL_SEEDS[domain]:
                raise ValueError(
                    f'{domain}: seeds {sorted(seeds)}, expected '
                    f'{sorted(EXPECTED_FORMAL_SEEDS[domain])}.'
                )
            for label, item in items:
                if (
                    int(item['manifest']['fold_idx']) != -1
                    or not item['summary']['validation']['exact_oof_coverage']
                    or item['manifest']['provenance']['purpose']
                    != 'formal_rebuttal'
                ):
                    raise ValueError(
                        f'{label}: partial/non-formal run cannot enter '
                        'headline seed aggregate.'
                    )
            reference_label, reference = items[0]
            for label, item in items[1:]:
                differences = comparison_manifest_differences(
                    reference['manifest'], item['manifest']
                )
                differences = [
                    key for key in differences if key not in {
                        'rand_seed', 'fold_seeds'
                    }
                ]
                if differences:
                    raise ValueError(
                        f'{reference_label}/{label}: seed runs differ in '
                        f'{differences}.'
                    )
        values = np.asarray([
            item['summary']['scores'][
                'retrospective_action_conditional'
            ]['valid_step_micro_mean']
            for _, item in items
        ])
        aggregate[domain] = {
            'labels': [label for label, _ in items],
            'seeds': seeds,
            'formal_complete': not allow_partial,
            'valid_step_micro_seed_mean': float(values.mean()),
            'valid_step_micro_seed_sample_sd': (
                float(values.std(ddof=1)) if len(values) > 1 else None
            ),
        }
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        action='append',
        required=True,
        type=parse_labeled_path,
        metavar='LABEL=PATH',
    )
    parser.add_argument(
        '--causal',
        action='append',
        default=[],
        type=parse_labeled_path,
        metavar='LABEL=PATH',
    )
    parser.add_argument(
        '--state-only',
        action='append',
        default=[],
        type=parse_labeled_path,
        metavar='LABEL=PATH',
    )
    parser.add_argument(
        '--historical',
        action='append',
        default=[],
        type=parse_labeled_path,
        metavar='LABEL=PATH',
    )
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='Permit fold smoke runs; excludes formal seed-set enforcement.',
    )
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    labels = [label for label, _ in args.experiment]
    if len(labels) != len(set(labels)):
        raise ValueError('Experiment labels must be unique.')
    causal_paths = dict(args.causal)
    if len(causal_paths) != len(args.causal):
        raise ValueError('Causal labels must be unique.')
    state_only_paths = dict(args.state_only)
    if len(state_only_paths) != len(args.state_only):
        raise ValueError('State-only labels must be unique.')
    historical_paths = dict(args.historical)
    if len(historical_paths) != len(args.historical):
        raise ValueError('Historical labels must be unique.')
    for name, paths in (
        ('Causal', causal_paths),
        ('State-only', state_only_paths),
        ('Historical', historical_paths),
    ):
        unknown = set(paths).difference(labels)
        if unknown:
            raise ValueError(
                f'{name} labels have no matching experiment: '
                f'{sorted(unknown)}.'
            )

    experiments = {
        label: load_energy_experiment(label, path)
        for label, path in args.experiment
    }
    report = {
        'schema': 'retrospective_conditional_energy_evaluation_v2',
        'experiments': {
            label: experiment['summary']
            for label, experiment in experiments.items()
        },
        'seed_aggregate': seed_aggregate(experiments, args.allow_partial),
        'pairwise_seed_stability': [
            stability
            for (left_label, left), (right_label, right) in itertools.combinations(
                experiments.items(), 2
            )
            for stability in [
                pairwise_stability(left_label, left, right_label, right)
            ]
            if stability is not None
        ],
        'predictive_baselines': {
            label: {
                **(
                    {
                        'causal': compare_with_predictive_baseline(
                            label, experiments[label], causal_paths[label], 'causal'
                        )
                    }
                    if label in causal_paths
                    else {}
                ),
                **(
                    {
                        'state_only': compare_with_predictive_baseline(
                            label,
                            experiments[label],
                            state_only_paths[label],
                            'state_only',
                        )
                    }
                    if label in state_only_paths
                    else {}
                ),
            }
            for label in labels
            if label in causal_paths or label in state_only_paths
        },
        'historical_recreated_segmentation': {
            label: compare_with_historical(
                label, experiments[label], path
            )
            for label, path in historical_paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as fout:
        json.dump(report, fout, indent=2)
        fout.write('\n')
    checksum_path = args.output.with_name(args.output.name + '.sha256')
    checksum_path.write_text(
        f'{sha256_file(args.output)}  {args.output.name}\n'
    )
    print(f'Wrote {args.output}')
    print(f'Wrote {checksum_path}')


if __name__ == '__main__':
    main()
