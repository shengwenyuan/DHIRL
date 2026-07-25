import os
import hashlib
import json
import argparse
import platform

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold

from src_autotest.algorithms_b import PGIAVI_B


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fin:
        for block in iter(lambda: fin.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pad_step_scores(step_scores, mask):
    padded = np.full(mask.shape, np.nan, dtype=np.float32)
    for idx, scores in enumerate(step_scores):
        padded[idx, :len(scores)] = scores
    return padded


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ll_filename', type=str, default='ll_pgiql.csv')
    parser.add_argument('--output_dir', type=str, default='outputs/labyrinth_train')
    parser.add_argument('--group_id', type=str, default='default')
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--num_latents', type=int, default=3)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='data/labyrinth/data')
    parser.add_argument('--gate_mode', type=str, default='retrospective',
                        choices=['retrospective', 'causal', 'state_only'])
    parser.add_argument('--fold_idx', type=int, default=-1,
                        help='Fold to run for smoke testing. -1 runs all folds.')
    parser.add_argument('--paired_fold_seeds', type=int, default=0,
                        help='Reset a deterministic seed before each fold. 1=on, 0=legacy stream.')
    parser.add_argument('--p0_artifacts', type=int, default=0,
                        help='Save explicit gate/responsibility/score arrays. 1=on, 0=legacy f arrays.')

    parser.add_argument('--model_type', type=str, default='IntentionRNN',
                        choices=['IntentionRNN', 'IntentionLSTM', 'IntentionTransformer'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--rnn_hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--reg_type', type=str, default='l1', choices=['l1', 'kl', 'kl+l1'])
    parser.add_argument('--reg_weight', type=float, default=0.35)
    parser.add_argument('--kl_weight', type=float, default=0.0)

    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--loss_threshold', type=float, default=5e-2)
    parser.add_argument('--max_iterations', type=int, default=120)

    args = parser.parse_args()

    num_folds = 5
    num_repeats = args.num_repeats
    num_states = 127
    num_actions = 4
    num_latents = args.num_latents
    if args.fold_idx < -1 or args.fold_idx >= num_folds:
        raise ValueError(f'fold_idx must be -1 or in [0, {num_folds - 1}].')

    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.rand_seed)

    run_dir = os.path.join(args.output_dir, args.group_id)
    os.makedirs(run_dir, exist_ok=True)
    output_df = pd.DataFrame(columns=[
        'num_trajs', 'fold', 'train_ll', 'test_ll', 'seed', 'fold_seed', 'gate_mode',
        'score_type', 'train_step_ll', 'test_step_ll', 'train_steps', 'test_steps',
        'iterations', 'stop_reason', 'final_loss', 'status',
    ])

    trans_path = os.path.join(args.data_dir, 'trans_probs.npy')
    trajs_path = os.path.join(args.data_dir, 'trajs.js')
    with open(trans_path, 'rb') as f:
        P = np.load(f)
    with open(trajs_path) as f:
        trajs = json.load(f)

    len_trajs = len(trajs)
    expected_p_shape = (num_states, num_actions, num_states)
    if P.shape != expected_p_shape:
        raise ValueError(f'Expected transition shape {expected_p_shape}, got {P.shape}.')
    if not np.isfinite(P).all() or (P < 0).any():
        raise ValueError('Transition tensor must be finite and nonnegative.')
    if not np.allclose(P.sum(axis=2), 1.0, atol=1e-6):
        raise ValueError('Labyrinth transition rows must sum to one.')
    if len_trajs < 238:
        raise ValueError(f'Expected at least 238 Labyrinth trajectories, got {len_trajs}.')
    if args.p0_artifacts and (
        len_trajs != 238 or any(len(traj) != 500 for traj in trajs)
    ):
        raise ValueError('P0 Labyrinth input must contain 238 trajectories of 500 steps.')
    manifest = {
        'domain': 'labyrinth',
        'gate_mode': args.gate_mode,
        'model_type': args.model_type,
        'num_latents': num_latents,
        'num_states': num_states,
        'num_actions': num_actions,
        'rand_seed': args.rand_seed,
        'num_repeats': num_repeats,
        'num_folds': num_folds,
        'fold_idx': args.fold_idx,
        'fold_random_state': 10042,
        'fold_seed_rule': (
            'rand_seed + 1009 * fold_idx + repeat_idx'
            if args.paired_fold_seeds
            else 'legacy continuous RNG stream'
        ),
        'fold_seeds': (
            [args.rand_seed + 1009 * idx for idx in range(num_folds)]
            if args.paired_fold_seeds
            else None
        ),
        'transition_shape': list(P.shape),
        'num_trajs_total': len_trajs,
        'num_trajs_used': 238,
        'num_steps_used': sum(len(traj) for traj in trajs[:238]),
        'score_type': (
            'retrospective_compatibility_score'
            if args.gate_mode == 'retrospective'
            else 'predictive_action_log_likelihood'
        ),
        'status': 'running',
        'transition_path': os.path.abspath(trans_path),
        'trajectory_path': os.path.abspath(trajs_path),
        'transition_bytes': os.path.getsize(trans_path),
        'trajectory_bytes': os.path.getsize(trajs_path),
        'transition_sha256': sha256_file(trans_path),
        'trajectory_sha256': sha256_file(trajs_path),
        'resolved_args': vars(args),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'device': str(device),
    }
    with open(os.path.join(run_dir, 'run_manifest.json'), 'w') as fout:
        json.dump(manifest, fout, indent=2)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10042)
    completed_folds = []
    for num_trajs in [238]:
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
            if args.fold_idx >= 0 and kf_idx != args.fold_idx:
                continue
            train_trajs = [trajs[train_idx] for train_idx in train_idxes]
            test_trajs = [trajs[test_idx] for test_idx in test_idxes]

            best_test_ll = -np.inf
            best_ll = None
            for repeats in range(num_repeats):
                model_seed = args.rand_seed + 1009 * kf_idx + repeats
                if args.paired_fold_seeds:
                    set_seed(model_seed)
                model = PGIAVI_B(
                    num_latents=num_latents, num_states=num_states, num_actions=num_actions,
                    train_trajs=train_trajs, test_trajs=test_trajs, P=P, discount=0.9,
                    model_type=args.model_type, hidden_dim=args.hidden_dim,
                    rnn_hidden_dim=args.rnn_hidden_dim, num_layers=args.num_layers,
                    dropout=args.dropout, nhead=args.nhead, lr=args.lr,
                    reg_type=args.reg_type, reg_weight=args.reg_weight, kl_weight=args.kl_weight,
                    num_epochs=args.num_epochs, loss_threshold=args.loss_threshold,
                    max_iterations=args.max_iterations, gate_mode=args.gate_mode,
                )
                ll, f, mask, agents = model.fit()
                if ll['test'] > best_test_ll:
                    best_test_ll = ll['test']
                    best_ll = ll
                    if num_trajs == len_trajs:
                        param_dir = os.path.join(run_dir, f'{num_trajs}/fold_{kf_idx}')
                        os.makedirs(param_dir, exist_ok=True)
                        with open(os.path.join(param_dir, 'train_idxes.json'), 'w') as fout:
                            json.dump(train_idxes.tolist(), fout)
                        with open(os.path.join(param_dir, 'test_idxes.json'), 'w') as fout:
                            json.dump(test_idxes.tolist(), fout)
                        np.save(os.path.join(param_dir, 'test_trajs.npy'), test_trajs)
                        np.save(os.path.join(param_dir, 'mask_train.npy'), mask['train'])
                        np.save(os.path.join(param_dir, 'mask_test.npy'), mask['test'])
                        if args.p0_artifacts:
                            np.save(os.path.join(param_dir, 'gate_train.npy'),
                                    np.asarray(f['gate_train']))
                            np.save(os.path.join(param_dir, 'gate_test.npy'),
                                    np.asarray(f['gate_test']))
                            np.save(os.path.join(param_dir, 'responsibility_train.npy'),
                                    np.asarray(f['responsibility_train']))
                            np.save(os.path.join(param_dir, 'responsibility_test.npy'),
                                    np.asarray(f['responsibility_test']))
                            np.save(os.path.join(param_dir, 'step_log_score_train.npy'),
                                    pad_step_scores(f['step_log_score_train'], mask['train']))
                            np.save(os.path.join(param_dir, 'step_log_score_test.npy'),
                                    pad_step_scores(f['step_log_score_test'], mask['test']))
                        else:
                            np.save(os.path.join(param_dir, 'f_train.npy'), f['train'])
                            np.save(os.path.join(param_dir, 'f_test.npy'), f['test'])
                        for agent_idx, agent in enumerate(agents):
                            np.save(os.path.join(param_dir, f'r_{agent_idx}.npy'), agent.r.cpu().numpy())
                            np.save(os.path.join(param_dir, f'q_{agent_idx}.npy'), agent.q.cpu().numpy())
            output_df.loc[len(output_df)] = [
                num_trajs, kf_idx, best_ll['train'], best_ll['test'], args.rand_seed,
                args.rand_seed + 1009 * kf_idx if args.paired_fold_seeds else np.nan,
                args.gate_mode,
                best_ll['score_type'], best_ll['train_step_mean'],
                best_ll['test_step_mean'], best_ll['train_steps'], best_ll['test_steps'],
                best_ll['iterations'], best_ll['stop_reason'], best_ll['final_loss'],
                'complete',
            ]
            output_df.to_csv(os.path.join(run_dir, args.ll_filename), index=False)
            completed_folds.append(kf_idx)

    manifest['status'] = 'complete'
    manifest['completed_folds'] = completed_folds
    with open(os.path.join(run_dir, 'run_manifest.json'), 'w') as fout:
        json.dump(manifest, fout, indent=2)
