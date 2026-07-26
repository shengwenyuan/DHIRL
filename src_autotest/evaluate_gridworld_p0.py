"""Evaluate P0 Gridworld artifacts without re-running inference or ``predict()``.

The component permutation is estimated once per fold from training
post-action responsibilities and is then reused for both held-out outputs and
the recovered Q functions.
"""

import argparse
import csv
import hashlib
import json
import os
import re

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from env.collect_demo import policy_eval, value_iteration, vi_policy
from env.gridworld import GridWorld


ROLES = ('goal', 'return')
PROBABILITY_TOLERANCE = 1e-5
VALUE_TOLERANCE = 1e-10


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fin:
        for block in iter(lambda: fin.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path) as fin:
        return json.load(fin)


def discover_folds(experiment_dir):
    numeric_dirs = []
    for name in os.listdir(experiment_dir):
        path = os.path.join(experiment_dir, name)
        if name.isdigit() and os.path.isdir(path):
            if any(re.fullmatch(r'fold_\d+', child) for child in os.listdir(path)):
                numeric_dirs.append((int(name), path))
    if len(numeric_dirs) != 1:
        raise ValueError(
            'Expected exactly one <num_trajs>/fold_* tree below '
            f'{experiment_dir}, found {[item[0] for item in numeric_dirs]}.'
        )

    num_trajs, artifact_root = numeric_dirs[0]
    folds = []
    for name in os.listdir(artifact_root):
        match = re.fullmatch(r'fold_(\d+)', name)
        path = os.path.join(artifact_root, name)
        if match and os.path.isdir(path):
            folds.append((int(match.group(1)), path))
    folds.sort()
    if not folds:
        raise ValueError(f'No fold directories found below {artifact_root}.')
    if len({fold_idx for fold_idx, _ in folds}) != len(folds):
        raise ValueError('Duplicate fold indices were discovered.')
    return num_trajs, folds


def validate_indices(indices, name, num_trajs):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f'{name} indices must be one-dimensional.')
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f'{name} contains duplicate trajectory indices.')
    if np.any(indices < 0) or np.any(indices >= num_trajs):
        raise ValueError(f'{name} contains an index outside [0, {num_trajs}).')
    return indices


def validate_probability_rows(probabilities, mask, name):
    valid = probabilities[mask]
    if not np.all(np.isfinite(valid)):
        raise ValueError(f'{name} contains a non-finite valid probability.')
    if np.any(valid < -PROBABILITY_TOLERANCE):
        raise ValueError(f'{name} contains a negative valid probability.')
    if not np.allclose(
        valid.sum(axis=-1), 1.0, atol=PROBABILITY_TOLERANCE, rtol=0.0
    ):
        raise ValueError(f'{name} valid rows do not sum to one.')
    if np.any(np.abs(probabilities[~mask]) > PROBABILITY_TOLERANCE):
        raise ValueError(f'{name} padded rows must be zero.')


def load_split(fold_dir, split, canonical_latents, num_trajs):
    indices = validate_indices(
        read_json(os.path.join(fold_dir, f'{split}_idxes.json')),
        f'{fold_dir}/{split}',
        num_trajs,
    )
    gate = np.load(os.path.join(fold_dir, f'gate_{split}.npy'))
    responsibility = np.load(
        os.path.join(fold_dir, f'responsibility_{split}.npy')
    )
    mask_raw = np.load(os.path.join(fold_dir, f'mask_{split}.npy'))
    saved_latents = np.load(os.path.join(fold_dir, f'latent_{split}.npy'))
    step_scores = np.load(
        os.path.join(fold_dir, f'step_log_score_{split}.npy')
    )

    if gate.ndim != 3:
        raise ValueError(f'gate_{split} must have shape (trajectory, step, component).')
    if responsibility.shape != gate.shape:
        raise ValueError(f'{split} gate and responsibility shapes differ.')
    if mask_raw.shape != gate.shape[:2]:
        raise ValueError(f'{split} mask shape does not match probability arrays.')
    if saved_latents.shape != mask_raw.shape:
        raise ValueError(f'{split} saved latent shape does not match its mask.')
    if step_scores.shape != mask_raw.shape:
        raise ValueError(f'{split} step-score shape does not match its mask.')
    if gate.shape[0] != len(indices):
        raise ValueError(f'{split} array count does not match its index file.')
    if mask_raw.dtype != np.bool_:
        if not np.all(np.isin(mask_raw, (0, 1))):
            raise ValueError(f'{split} mask contains values other than zero or one.')
    mask = mask_raw.astype(bool)
    validate_probability_rows(gate, mask, f'gate_{split}')
    validate_probability_rows(
        responsibility, mask, f'responsibility_{split}'
    )
    if not np.isfinite(step_scores[mask]).all():
        raise ValueError(f'{split} has non-finite valid step scores.')
    if not np.isnan(step_scores[~mask]).all():
        raise ValueError(f'{split} padded step scores must be NaN.')

    labels = []
    gates = []
    responsibilities = []
    scores = []
    for row, trajectory_idx in enumerate(indices):
        truth = np.asarray(canonical_latents[trajectory_idx], dtype=np.int64)
        expected_mask = np.arange(mask.shape[1]) < len(truth)
        if not np.array_equal(mask[row], expected_mask):
            raise ValueError(
                f'{split} mask for trajectory {trajectory_idx} is not the '
                'canonical prefix length.'
            )
        if not np.array_equal(saved_latents[row, expected_mask], truth):
            raise ValueError(
                f'{split} saved labels disagree with canonical labels for '
                f'trajectory {trajectory_idx}.'
            )
        labels.append(truth)
        gates.append(np.asarray(gate[row, expected_mask], dtype=np.float64))
        responsibilities.append(
            np.asarray(responsibility[row, expected_mask], dtype=np.float64)
        )
        scores.append(
            np.asarray(step_scores[row, expected_mask], dtype=np.float64)
        )

    return {
        'indices': indices,
        'labels': labels,
        'gate': gates,
        'responsibility': responsibilities,
        'step_score': scores,
        'num_components': gate.shape[-1],
    }


def fit_train_mapping(labels, responsibilities, num_components):
    label_values = np.unique(np.concatenate(labels))
    expected_labels = np.arange(num_components)
    if not np.array_equal(label_values, expected_labels):
        raise ValueError(
            f'Expected canonical labels {expected_labels.tolist()}, '
            f'found {label_values.tolist()}.'
        )

    soft_confusion = np.zeros((num_components, num_components), dtype=np.float64)
    label_counts = np.zeros(num_components, dtype=np.int64)
    for truth, probability in zip(labels, responsibilities):
        for label in expected_labels:
            selected = truth == label
            soft_confusion[:, label] += probability[selected].sum(axis=0)
            label_counts[label] += int(selected.sum())
    if np.any(label_counts == 0):
        raise ValueError('Every canonical label must occur in the training fold.')

    raw_components, canonical_labels = linear_sum_assignment(-soft_confusion)
    mapping = np.full(num_components, -1, dtype=np.int64)
    mapping[raw_components] = canonical_labels
    if not np.array_equal(np.sort(mapping), expected_labels):
        raise ValueError('Hungarian assignment did not produce a permutation.')
    normalized = soft_confusion / label_counts[np.newaxis, :]
    return mapping, soft_confusion, normalized, label_counts


def align_probabilities(probabilities, mapping):
    aligned = np.empty_like(probabilities)
    for raw_component, canonical_label in enumerate(mapping):
        aligned[:, canonical_label] = probabilities[:, raw_component]
    return aligned


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def matched_boundary_counts(truth_sequences, predicted_sequences, tolerance):
    true_count = 0
    predicted_count = 0
    matched_count = 0
    for truth, predicted in zip(truth_sequences, predicted_sequences):
        truth = np.asarray(truth)
        predicted = np.asarray(predicted)
        true_boundaries = np.flatnonzero(truth[1:] != truth[:-1]) + 1
        predicted_boundaries = np.flatnonzero(
            predicted[1:] != predicted[:-1]
        ) + 1
        true_count += len(true_boundaries)
        predicted_count += len(predicted_boundaries)

        true_idx = 0
        predicted_idx = 0
        while (
            true_idx < len(true_boundaries)
            and predicted_idx < len(predicted_boundaries)
        ):
            distance = predicted_boundaries[predicted_idx] - true_boundaries[true_idx]
            if abs(distance) <= tolerance:
                matched_count += 1
                true_idx += 1
                predicted_idx += 1
            elif distance < -tolerance:
                predicted_idx += 1
            else:
                true_idx += 1
    return true_count, predicted_count, matched_count


def segmentation_metrics(label_sequences, probability_sequences):
    truth = np.concatenate(label_sequences)
    probabilities = np.concatenate(probability_sequences)
    predicted_sequences = [
        probability.argmax(axis=-1) for probability in probability_sequences
    ]
    predicted = np.concatenate(predicted_sequences)
    num_classes = probabilities.shape[-1]
    one_hot = np.eye(num_classes, dtype=np.float64)[truth]
    selected_probability = np.clip(
        probabilities[np.arange(len(truth)), truth], 1e-15, 1.0
    )

    metrics = {
        'num_steps': int(len(truth)),
        'balanced_accuracy': float(balanced_accuracy_score(truth, predicted)),
        'macro_f1': float(
            f1_score(
                truth,
                predicted,
                labels=np.arange(num_classes),
                average='macro',
                zero_division=0,
            )
        ),
        'multiclass_log_loss': float(-np.mean(np.log(selected_probability))),
        'multiclass_brier': float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=-1))
        ),
        'support_by_class': np.bincount(
            truth, minlength=num_classes
        ).astype(int).tolist(),
        'hard_confusion_true_by_predicted': confusion_matrix(
            truth, predicted, labels=np.arange(num_classes)
        ).astype(int).tolist(),
    }
    for tolerance, suffix in ((0, 'exact'), (1, 'tol1')):
        true_count, predicted_count, matched_count = matched_boundary_counts(
            label_sequences, predicted_sequences, tolerance
        )
        precision = safe_ratio(matched_count, predicted_count)
        recall = safe_ratio(matched_count, true_count)
        metrics.update({
            f'boundary_{suffix}_true': int(true_count),
            f'boundary_{suffix}_predicted': int(predicted_count),
            f'boundary_{suffix}_matched': int(matched_count),
            f'boundary_{suffix}_precision': precision,
            f'boundary_{suffix}_recall': recall,
            f'boundary_{suffix}_f1': safe_ratio(
                2.0 * precision * recall, precision + recall
            ),
        })
    return metrics


def softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def score_metrics(score_sequences):
    return {
        'trajectory_macro_mean': float(np.mean([
            scores.mean() for scores in score_sequences
        ])),
        'valid_step_micro_mean': float(
            sum(scores.sum() for scores in score_sequences)
            / sum(len(scores) for scores in score_sequences)
        ),
        'num_valid_steps': int(sum(len(scores) for scores in score_sequences)),
    }


def empirical_state_weights(indices, trajectories, latents, num_labels, num_states):
    counts = np.zeros((num_labels, num_states), dtype=np.float64)
    for trajectory_idx in indices:
        for transition, label in zip(
            trajectories[trajectory_idx], latents[trajectory_idx]
        ):
            counts[int(label), int(transition[0])] += 1.0
    totals = counts.sum(axis=1)
    if (totals == 0).any():
        raise ValueError('Every canonical role needs test-fold state support.')
    return counts / totals[:, None]


def canonical_control_objects(env):
    rewards = []
    rewards.append(np.zeros(env.num_states, dtype=np.float64))
    rewards[-1][env.state_to_int(env.goal_state)] = 1.0
    rewards.append(np.zeros(env.num_states, dtype=np.float64))
    rewards[-1][env.state_to_int(env.initial_state)] = 1.0

    objects = []
    for role, reward in zip(ROLES, rewards):
        optimal_value = value_iteration(
            reward,
            env.P,
            env.num_states,
            env.num_actions,
            env.gamma,
            threshold=VALUE_TOLERANCE,
        )
        optimal_policy = vi_policy(
            env.num_states,
            env.num_actions,
            env.P,
            reward,
            env.gamma,
            stochastic=False,
            threshold=VALUE_TOLERANCE,
        )
        objects.append({
            'role': role,
            'reward': reward,
            'optimal_value': optimal_value,
            'optimal_policy': optimal_policy,
        })
    return objects


def policy_metrics(fold_dir, mapping, env, canonical_objects, state_weights):
    if len(mapping) != len(canonical_objects):
        raise ValueError(
            f'Expected {len(canonical_objects)} components, got {len(mapping)}.'
        )
    initial_state = env.state_to_int(env.initial_state)
    by_role = {}
    for canonical_label, canonical in enumerate(canonical_objects):
        raw_component = int(np.flatnonzero(mapping == canonical_label)[0])
        q_path = os.path.join(fold_dir, f'q_{raw_component}.npy')
        q = np.load(q_path)
        expected_shape = (env.num_states, env.num_actions)
        if q.shape != expected_shape:
            raise ValueError(
                f'{q_path} has shape {q.shape}; expected {expected_shape}.'
            )
        if not np.all(np.isfinite(q)):
            raise ValueError(f'{q_path} contains non-finite values.')

        learned_policy = softmax(np.asarray(q, dtype=np.float64))
        learned_greedy = learned_policy.argmax(axis=-1)
        optimal_greedy = canonical['optimal_policy'].argmax(axis=-1)
        learned_value = policy_eval(
            learned_policy,
            canonical['reward'],
            env.P,
            env.num_states,
            env.gamma,
            threshold=VALUE_TOLERANCE,
        )
        optimal_value = canonical['optimal_value']
        by_role[canonical['role']] = {
            'canonical_label': canonical_label,
            'raw_component': raw_component,
            'greedy_action_agreement': float(
                np.mean(learned_greedy == optimal_greedy)
            ),
            'mean_abs_value_difference': float(
                np.mean(np.abs(optimal_value - learned_value))
            ),
            'empirical_state_weighted_abs_value_difference': float(
                np.dot(
                    state_weights[canonical_label],
                    np.abs(optimal_value - learned_value),
                )
            ),
            'initial_state_regret_vstar_minus_vpi': float(
                optimal_value[initial_state] - learned_value[initial_state]
            ),
        }

    macro = {}
    for metric in (
        'greedy_action_agreement',
        'mean_abs_value_difference',
        'empirical_state_weighted_abs_value_difference',
        'initial_state_regret_vstar_minus_vpi',
    ):
        macro[metric] = float(
            np.mean([by_role[role][metric] for role in ROLES])
        )
    return {'by_role': by_role, 'macro': macro}


def scalar_metric_items(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def fold_csv_row(fold_summary):
    row = {
        'row_type': 'fold',
        'fold': fold_summary['fold'],
        'num_trajs': fold_summary['num_trajs'],
        'num_train_trajs': fold_summary['num_train_trajs'],
        'num_test_trajs': fold_summary['num_test_trajs'],
        'mapping_raw_to_canonical': json.dumps(
            fold_summary['mapping_raw_to_canonical'], separators=(',', ':')
        ),
    }
    for source in ('gate_prior', 'post_action_responsibility'):
        for key, value in scalar_metric_items(
            fold_summary['segmentation'][source]
        ).items():
            row[f'{source}_{key}'] = value
    for key, value in fold_summary['step_score'].items():
        row[f'step_score_{key}'] = value
    for role in ROLES:
        for key, value in fold_summary['policy']['by_role'][role].items():
            if isinstance(value, (int, float)):
                row[f'policy_{role}_{key}'] = value
    for key, value in fold_summary['policy']['macro'].items():
        row[f'policy_macro_{key}'] = value
    return row


def oof_csv_row(num_trajs, oof_summary, policy_fold_summary):
    row = {
        'row_type': 'oof',
        'fold': 'OOF',
        'num_trajs': num_trajs,
        'num_train_trajs': '',
        'num_test_trajs': num_trajs,
        'mapping_raw_to_canonical': 'fold-specific train-only mappings',
    }
    for source in ('gate_prior', 'post_action_responsibility'):
        for key, value in scalar_metric_items(oof_summary[source]).items():
            row[f'{source}_{key}'] = value
    for key, value in oof_summary['step_score'].items():
        if isinstance(value, (int, float)):
            row[f'step_score_{key}'] = value
    for role in ROLES:
        for key, value in policy_fold_summary['by_role'][role]['mean'].items():
            row[f'policy_{role}_{key}'] = value
    for key, value in policy_fold_summary['macro']['mean'].items():
        row[f'policy_macro_{key}'] = value
    return row


def summarize_policy_folds(fold_summaries):
    summary = {'by_role': {}, 'macro': {}}
    metric_names = (
        'greedy_action_agreement',
        'mean_abs_value_difference',
        'empirical_state_weighted_abs_value_difference',
        'initial_state_regret_vstar_minus_vpi',
    )
    for role in ROLES:
        values = {
            metric: np.asarray([
                fold['policy']['by_role'][role][metric]
                for fold in fold_summaries
            ])
            for metric in metric_names
        }
        summary['by_role'][role] = {
            'mean': {
                metric: float(metric_values.mean())
                for metric, metric_values in values.items()
            },
            'std': {
                metric: float(metric_values.std(ddof=0))
                for metric, metric_values in values.items()
            },
        }
    values = {
        metric: np.asarray([
            fold['policy']['macro'][metric] for fold in fold_summaries
        ])
        for metric in metric_names
    }
    summary['macro'] = {
        'mean': {
            metric: float(metric_values.mean())
            for metric, metric_values in values.items()
        },
        'std': {
            metric: float(metric_values.std(ddof=0))
            for metric, metric_values in values.items()
        },
    }
    return summary


def write_csv(path, rows):
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    experiment_dir = os.path.abspath(args.experiment_dir)
    data_dir = os.path.abspath(args.data_dir)
    num_trajs, folds = discover_folds(experiment_dir)
    latent_path = os.path.join(data_dir, 'latents_frustration.json')
    trajectory_path = os.path.join(data_dir, 'trajs_frustration.json')
    canonical_latents = read_json(latent_path)
    canonical_trajectories = read_json(trajectory_path)
    if len(canonical_trajectories) != len(canonical_latents):
        raise ValueError('Canonical trajectories and latent labels differ in count.')
    if len(canonical_latents) < num_trajs:
        raise ValueError(
            f'Canonical file has {len(canonical_latents)} trajectories, '
            f'but artifacts require {num_trajs}.'
        )
    canonical_latents = canonical_latents[:num_trajs]
    canonical_trajectories = canonical_trajectories[:num_trajs]
    if any(len(labels) == 0 for labels in canonical_latents):
        raise ValueError('Canonical latent trajectories must be non-empty.')
    if any(
        len(trajectory) != len(labels)
        for trajectory, labels in zip(canonical_trajectories, canonical_latents)
    ):
        raise ValueError('Canonical trajectories and labels are not step-aligned.')

    manifest_path = os.path.join(experiment_dir, 'run_manifest.json')
    if not os.path.isfile(manifest_path):
        raise ValueError(f'Missing required run manifest: {manifest_path}')
    manifest = read_json(manifest_path)
    used = manifest.get('num_trajs_used')
    if used is None or int(used) != num_trajs:
        raise ValueError(
            f'Manifest num_trajs_used={used} disagrees with {num_trajs}/.'
        )
    expected_hashes = {
        'trajectory_sha256': sha256_file(trajectory_path),
        'latent_sha256': sha256_file(latent_path),
    }
    for key, actual_hash in expected_hashes.items():
        if manifest.get(key) != actual_hash:
            raise ValueError(
                f'Manifest {key} does not match the canonical input.'
            )

    env = GridWorld()
    canonical_objects = canonical_control_objects(env)
    coverage = np.zeros(num_trajs, dtype=np.int64)
    oof_gate = [None] * num_trajs
    oof_responsibility = [None] * num_trajs
    oof_step_score = [None] * num_trajs
    fold_summaries = []

    for fold_idx, fold_dir in folds:
        train = load_split(
            fold_dir, 'train', canonical_latents, num_trajs
        )
        test = load_split(
            fold_dir, 'test', canonical_latents, num_trajs
        )
        if train['num_components'] != test['num_components']:
            raise ValueError(f'Fold {fold_idx} train/test component counts differ.')
        if set(train['indices']) & set(test['indices']):
            raise ValueError(f'Fold {fold_idx} train and test indices overlap.')
        if set(np.concatenate((train['indices'], test['indices']))) != set(
            range(num_trajs)
        ):
            raise ValueError(f'Fold {fold_idx} train/test indices are not exhaustive.')

        mapping, soft_confusion, normalized_confusion, label_counts = (
            fit_train_mapping(
                train['labels'],
                train['responsibility'],
                train['num_components'],
            )
        )
        aligned_gate = [
            align_probabilities(probability, mapping)
            for probability in test['gate']
        ]
        aligned_responsibility = [
            align_probabilities(probability, mapping)
            for probability in test['responsibility']
        ]
        segmentation = {
            'gate_prior': segmentation_metrics(test['labels'], aligned_gate),
            'post_action_responsibility': segmentation_metrics(
                test['labels'], aligned_responsibility
            ),
        }
        state_weights = empirical_state_weights(
            test['indices'], canonical_trajectories, canonical_latents,
            train['num_components'], env.num_states,
        )
        control = policy_metrics(
            fold_dir, mapping, env, canonical_objects, state_weights
        )
        fold_summary = {
            'fold': fold_idx,
            'num_trajs': num_trajs,
            'num_train_trajs': int(len(train['indices'])),
            'num_test_trajs': int(len(test['indices'])),
            'num_train_steps': int(sum(map(len, train['labels']))),
            'num_test_steps': int(sum(map(len, test['labels']))),
            'mapping_raw_to_canonical': mapping.astype(int).tolist(),
            'train_label_counts': label_counts.astype(int).tolist(),
            'train_soft_confusion_raw_by_true': soft_confusion.tolist(),
            'train_soft_confusion_per_true_step': normalized_confusion.tolist(),
            'segmentation': segmentation,
            'step_score': {
                'score_type': (
                    manifest.get('score_type') if manifest is not None else None
                ),
                **score_metrics(test['step_score']),
            },
            'policy': control,
        }
        fold_summaries.append(fold_summary)

        for row, trajectory_idx in enumerate(test['indices']):
            coverage[trajectory_idx] += 1
            if oof_gate[trajectory_idx] is not None:
                raise ValueError(
                    f'Trajectory {trajectory_idx} appears in multiple test folds.'
                )
            oof_gate[trajectory_idx] = aligned_gate[row]
            oof_responsibility[trajectory_idx] = aligned_responsibility[row]
            oof_step_score[trajectory_idx] = test['step_score'][row]

    if not np.all(coverage == 1):
        missing = np.flatnonzero(coverage == 0).tolist()
        repeated = np.flatnonzero(coverage > 1).tolist()
        raise ValueError(
            'OOF coverage must contain each trajectory exactly once; '
            f'missing={missing}, repeated={repeated}.'
        )
    oof = {
        'coverage_min': int(coverage.min()),
        'coverage_max': int(coverage.max()),
        'num_trajectories': num_trajs,
        'gate_prior': segmentation_metrics(canonical_latents, oof_gate),
        'post_action_responsibility': segmentation_metrics(
            canonical_latents, oof_responsibility
        ),
        'step_score': {
            'score_type': (
                manifest.get('score_type') if manifest is not None else None
            ),
            **score_metrics(oof_step_score),
        },
    }
    policy_fold_summary = summarize_policy_folds(fold_summaries)

    csv_rows = [fold_csv_row(fold) for fold in fold_summaries]
    csv_rows.append(oof_csv_row(num_trajs, oof, policy_fold_summary))
    write_csv(args.output_csv, csv_rows)

    summary = {
        'schema_version': 1,
        'experiment_dir': experiment_dir,
        'data_dir': data_dir,
        'canonical_latent_path': os.path.abspath(latent_path),
        'canonical_trajectory_path': os.path.abspath(trajectory_path),
        'manifest': manifest,
        'num_trajs': num_trajs,
        'fold_indices': [fold_idx for fold_idx, _ in folds],
        'canonical_roles': list(ROLES),
        'assumptions': [
            'Canonical latent 0 is goal and latent 1 is return, matching '
            'env.collect_demo.collect_frustrationcase.',
            'The raw-to-canonical permutation maximizes summed training '
            'post-action responsibility mass; test labels never affect it.',
            'Gate prior and post-action responsibility are evaluated and '
            'reported separately under the same fold mapping.',
            'Classification metrics pool valid held-out steps; Brier is the '
            'mean sum of squared multiclass probability errors.',
            'A boundary is the first step of a new segment. Exact and +/-1 '
            'scores aggregate ordered one-to-one matches within trajectories.',
            'Recovered policies are softmax(Q). Greedy agreement uses argmax, '
            'while value metrics evaluate the full softmax policy.',
            'Value MAE is mean |V* - Vpi|; initial-state regret retains the '
            'direction V*(s0) - Vpi(s0).',
            'Empirical state-weighted value difference uses held-out state '
            'frequencies for the corresponding canonical latent role.',
        ],
        'folds': fold_summaries,
        'oof': oof,
        'policy_fold_summary': policy_fold_summary,
        'output_csv': os.path.abspath(args.output_csv),
    }
    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w') as fout:
        json.dump(summary, fout, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate saved P0 Gridworld folds without model inference.'
    )
    parser.add_argument('--experiment-dir', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-csv', required=True)
    args = parser.parse_args()
    summary = evaluate(args)
    print(
        f"Evaluated {len(summary['folds'])} folds and "
        f"{summary['oof']['num_trajectories']} OOF trajectories."
    )
    print(f"JSON: {os.path.abspath(args.output_json)}")
    print(f"CSV:  {os.path.abspath(args.output_csv)}")


if __name__ == '__main__':
    main()
